from flipside import Flipside
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnchoredOffsetbox
import matplotlib.image as mpimg
import os
import io
from anthropic import Anthropic
import textwrap
import base64


def format_number(num):
    if num >= 1_000_000:
        return f"{num/1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num/1_000:.2f}K"
    else:
        return f"{num:.2f}"

def millions_formatter(x, pos):
    return f'{x/1_000_000:.1f}M' if x >= 1_000_000 else f'{x/1_000:.1f}K' if x >= 1_000 else f'{x:.0f}'

def create_plot_with_background(fig, ax, title, timestamp, filename, zoom_factor):
    # Make sure charts directory exists
    charts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'charts')
    if not os.path.exists(charts_dir):
        os.makedirs(charts_dir)
    
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    background_path = os.path.join(script_dir, 'background.png')
    
    if not os.path.exists(background_path):
        print(f"Warning: background.png not found at {background_path}")
        plt.savefig(os.path.join(charts_dir, f'{filename}_{timestamp}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close('all')
        return
    
    # Read background image
    background = mpimg.imread(background_path)
    
    # Save current plot to a buffer with transparency
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=300, transparent=True)
    buf.seek(0)
    plot_image = mpimg.imread(buf)
    
    # Create new figure with background
    fig_final = plt.figure(figsize=(16, 9))
    ax_final = fig_final.add_subplot(111)
    
    # Display background
    ax_final.imshow(background)
    
    plot_box = OffsetImage(plot_image, zoom=zoom_factor)
    anchored_box = AnchoredOffsetbox(
        loc='center',
        child=plot_box,
        pad=0,
        frameon=False,
        bbox_to_anchor=(0.5, 0.5),
        bbox_transform=ax_final.transAxes
    )
    ax_final.add_artist(anchored_box)
    
    # Remove axes and margins
    ax_final.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # Save with transparency preserved
    output_path = os.path.join(charts_dir, f'{filename}_{timestamp}.png')
    plt.savefig(output_path, 
                dpi=300, 
                bbox_inches='tight', 
                pad_inches=0,
                facecolor='none',
                edgecolor='none')
    plt.close('all')
    buf.close()

def get_voting_analysis(api_key, prop_id, vote_choices):
    # Set global plot styling for transparency
    plt.style.use('default')
    plt.rcParams.update({
        'figure.facecolor': 'none',
        'axes.facecolor': 'none',
        'savefig.facecolor': 'none'
    })
    
    flipside = Flipside(api_key, "https://api-v2.flipsidecrypto.xyz")
    
    # Hourly analysis query
    hourly_sql = f"""
    WITH vote_counts AS (
      SELECT ins.*
      FROM solana.core.fact_decoded_instructions ins 
      WHERE ins.program_id = 'GovaE4iu227srtG2s3tZzB4RmWBzw8sTwrCLZz7kN7rY' 
      AND ins.event_type = 'setVote' 
      AND ins.decoded_instruction:args:weight / pow(10, 6) is not null 
    ),
    HourlyVotes AS (
      SELECT 
        date_trunc('hour', block_timestamp) as hour,
        case 
          {" ".join(f"when decoded_instruction['args']['side'] like {num} then '{choice}'" for num, choice in vote_choices.items())}
        end as vote_choice,
        count(distinct signers[0]) as voters,
        sum(decoded_instruction['args']['weight'] / power(10, 6)) as voting_power
      FROM vote_counts
      WHERE DECODED_INSTRUCTION['accounts'][1]['pubkey'] = '{prop_id}'
      GROUP BY 1, 2
      ORDER BY 1
    )
    SELECT * FROM HourlyVotes
    """

    # Lifecycle and ranking comparison query
    lifecycle_sql = f"""
    WITH vote_counts AS (
      SELECT ins.*
      FROM solana.core.fact_decoded_instructions ins 
      WHERE ins.program_id = 'GovaE4iu227srtG2s3tZzB4RmWBzw8sTwrCLZz7kN7rY' 
      AND ins.event_type = 'setVote' 
      AND ins.decoded_instruction:args:weight / pow(10, 6) is not null 
    ), 
    CurrentPropLength AS (
      SELECT 
        DATEDIFF('hour', min(block_timestamp), max(block_timestamp)) as prop_length
      FROM vote_counts
      WHERE DECODED_INSTRUCTION['accounts'][1]['pubkey'] = '{prop_id}'
    ),
    CurrentProp AS (
      SELECT 
        signers[0] as wallet,
        case 
          {" ".join(f"when decoded_instruction['args']['side'] like {num} then '{choice}'" for num, choice in vote_choices.items())}
        end as vote_choice,
        decoded_instruction['args']['weight'] / power(10, 6) as voting_power
      FROM vote_counts
      WHERE DECODED_INSTRUCTION['accounts'][1]['pubkey'] = '{prop_id}'
    ),
    CurrentPropMetrics AS (
      SELECT 
        vote_choice,
        count(distinct wallet) as voters,
        sum(voting_power) as total_power
      FROM CurrentProp
      GROUP BY 1
    ),
    TotalCurrentMetrics AS (
      SELECT
        sum(voters) as total_voters,
        sum(total_power) as total_power
      FROM CurrentPropMetrics
    ),
    OtherPropsMetrics AS (
      SELECT
        vc.DECODED_INSTRUCTION['accounts'][1]['pubkey'] as prop_id,
        count(distinct signers[0]) as voters,
        sum(vc.decoded_instruction:args:weight / power(10, 6)) as voting_power
      FROM vote_counts vc
      WHERE vc.DECODED_INSTRUCTION['accounts'][1]['pubkey'] != '{prop_id}'
      GROUP BY 1
      HAVING voters > 10
    ),
    ComparisonStats AS (
      SELECT
        count(*) as total_props,
        avg(voting_power) as avg_power,
        avg(voters) as avg_voters,
        percentile_cont(0.5) within group (order by voting_power) as median_power,
        percentile_cont(0.5) within group (order by voters) as median_voters,
        max(voting_power) as max_power,
        max(voters) as max_voters,
        min(voting_power) as min_power,
        min(voters) as min_voters,
        (SELECT total_power FROM TotalCurrentMetrics) as current_power,
        (SELECT total_voters FROM TotalCurrentMetrics) as current_voters,
        count(case when voting_power > (SELECT total_power FROM TotalCurrentMetrics) then 1 end) as props_with_more_power,
        count(case when voters > (SELECT total_voters FROM TotalCurrentMetrics) then 1 end) as props_with_more_voters
      FROM OtherPropsMetrics
    ),
    Rankings AS (
      SELECT
        comp.*,
        props_with_more_power + 1 as power_rank,
        props_with_more_voters + 1 as voter_rank
      FROM ComparisonStats comp
    ),
    CurrentVotes AS (
      SELECT 
        c.*,
        r.*
      FROM CurrentPropMetrics c
      CROSS JOIN Rankings r
    )
    SELECT * FROM CurrentVotes
    """

    # Vote size distribution query
    size_sql = f"""
    WITH vote_counts AS (
      SELECT ins.*
      FROM solana.core.fact_decoded_instructions ins 
      WHERE ins.program_id = 'GovaE4iu227srtG2s3tZzB4RmWBzw8sTwrCLZz7kN7rY' 
      AND ins.event_type = 'setVote' 
      AND ins.decoded_instruction:args:weight / pow(10, 6) is not null 
    ),
    RankedVotes AS (
      SELECT 
        signers[0] as wallet,
        case 
          {" ".join(f"when decoded_instruction['args']['side'] like {num} then '{choice}'" for num, choice in vote_choices.items())}
        end as vote_choice,
        decoded_instruction['args']['weight'] / power(10, 6) as voting_power
      FROM vote_counts
      WHERE DECODED_INSTRUCTION['accounts'][1]['pubkey'] = '{prop_id}'
    ),
    VoteSizeAnalysis AS (
      SELECT
        vote_choice,
        power_group,
        COUNT(*) as num_wallets,
        SUM(voting_power) as total_voting_power
      FROM (
        SELECT 
          wallet,
          vote_choice,
          voting_power,
          CASE 
            WHEN voting_power < 10 THEN 'a/ Below 10'
            WHEN voting_power < 100 THEN 'b/ 10-100'
            WHEN voting_power < 1000 THEN 'c/ 100-1K'
            WHEN voting_power < 10000 THEN 'd/ 1K-10K'
            WHEN voting_power < 100000 THEN 'e/ 10K-100K'
            WHEN voting_power < 1000000 THEN 'f/ 100K-1M'
            WHEN voting_power < 10000000 THEN 'g/ 1M-10M'
            ELSE 'h/ 10M+'
          END as power_group
        FROM RankedVotes
      )
      GROUP BY 1, 2
    ),
    TotalMetrics AS (
      SELECT
        vote_choice,
        count(*) as total_voters,
        sum(voting_power) as total_power
      FROM RankedVotes
      GROUP BY 1
    )
    SELECT 
      v.*,
      t.total_voters,
      t.total_power
    FROM VoteSizeAnalysis v
    JOIN TotalMetrics t ON t.vote_choice = v.vote_choice
    ORDER BY 
      v.vote_choice,
      v.power_group
    """

    # Execute all queries
    hourly_results = flipside.query(hourly_sql)
    lifecycle_results = flipside.query(lifecycle_sql)
    size_results = flipside.query(size_sql)
    
    hourly_df = pd.DataFrame([r for r in hourly_results.records if r])
    lifecycle_df = pd.DataFrame([r for r in lifecycle_results.records if r])
    size_df = pd.DataFrame([r for r in size_results.records if r])
    
    # Create visualizations
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 1. Hourly Voting Power Graph
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_facecolor('none')
    fig.patch.set_alpha(0.0)
    
    # Pre-process datetime for the entire dataframe
    hourly_df['hour'] = pd.to_datetime(hourly_df['hour'])
    max_hours = int((hourly_df['hour'].max() - hourly_df['hour'].min()).total_seconds() / 3600)
    
    for choice in vote_choices.values():
        choice_data = hourly_df[hourly_df['vote_choice'] == choice].copy()
        choice_data = choice_data.sort_values('hour')
        
        if not choice_data.empty:
            start_time = choice_data['hour'].min()
            choice_data['hours'] = ((choice_data['hour'] - start_time).dt.total_seconds() / 3600).astype(int)
            
            ax.plot(choice_data['hours'], 
                   choice_data['voting_power'].cumsum(), 
                   label=choice, 
                   marker='o', 
                   markersize=4,
                   linewidth=2)
    
    ax.set_title('Cumulative Voting Power Over Time', pad=20, fontsize=12, color='black')
    ax.set_xlabel('Hours Since Start', fontsize=10, color='black')
    ax.set_ylabel('Voting Power', fontsize=10, color='black')
    
    if not hourly_df.empty:
        tick_spacing = max(1, max_hours // 10)
        ax.set_xticks(np.arange(0, max_hours + 1, tick_spacing))
        
    ax.tick_params(colors='black')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.yaxis.set_major_formatter(FuncFormatter(millions_formatter))
    ax.grid(True, alpha=0.3, color='gray')
    plt.tight_layout()
    
    create_plot_with_background(fig, ax, 'Cumulative Voting Power Over Time', 
                              timestamp, 'voting_power_trend',
                              zoom_factor=0.35)
    plt.close()

    # 2. Hourly Voters Graph
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_facecolor('none')
    fig.patch.set_alpha(0.0)
    
    for choice in vote_choices.values():
        choice_data = hourly_df[hourly_df['vote_choice'] == choice].copy()
        choice_data = choice_data.sort_values('hour')
        
        if not choice_data.empty:
            start_time = choice_data['hour'].min()
            choice_data['hours'] = ((choice_data['hour'] - start_time).dt.total_seconds() / 3600).astype(int)
            
            ax.plot(choice_data['hours'], 
                   choice_data['voters'].cumsum(), 
                   label=choice, 
                   marker='o', 
                   markersize=4,
                   linewidth=2)
    
    ax.set_title('Cumulative Voters Over Time', pad=20, fontsize=12, color='black')
    ax.set_xlabel('Hours Since Start', fontsize=10, color='black')
    ax.set_ylabel('Number of Voters', fontsize=10, color='black')
    
    if not hourly_df.empty:
        tick_spacing = max(1, max_hours // 10)
        ax.set_xticks(np.arange(0, max_hours + 1, tick_spacing))
        ax.tick_params(colors='black')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, color='gray')
    plt.tight_layout()
    
    create_plot_with_background(fig, ax, 'Cumulative Voters Over Time', 
                              timestamp, 'voter_trend',
                              zoom_factor=0.35)
    plt.close()

    # 3. Comparison with Other Proposals
    if not lifecycle_df.empty:
        first_row = lifecycle_df.iloc[0]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.set_facecolor('none')
        ax2.set_facecolor('none')
        fig.patch.set_alpha(0.0)
        
        # Voting Power Comparison
        comparison_data = {
            'Current': float(first_row['current_power']),
            'Average': float(first_row['avg_power']),
            'Median': float(first_row['median_power']),
            'Max': float(first_row['max_power'])
        }
        bars1 = ax1.bar(comparison_data.keys(), comparison_data.values())
        ax1.set_title('Voting Power Comparison', color='black')
        ax1.yaxis.set_major_formatter(FuncFormatter(millions_formatter))
        ax1.grid(True, alpha=0.3, color='gray')
        ax1.tick_params(colors='black')
        
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    millions_formatter(height, 0),
                    ha='center', va='bottom',
                    color='black')
        
        # Voter Count Comparison
        voter_data = {
            'Current': float(first_row['current_voters']),
            'Average': float(first_row['avg_voters']),
            'Median': float(first_row['median_voters']),
            'Max': float(first_row['max_voters'])
        }
        bars2 = ax2.bar(voter_data.keys(), voter_data.values())
        ax2.set_title('Voter Count Comparison', color='black')
        ax2.grid(True, alpha=0.3, color='gray')
        ax2.tick_params(colors='black')
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:,.0f}',
                    ha='center', va='bottom',
                    color='black')
        
        plt.tight_layout()
        create_plot_with_background(fig, (ax1, ax2), 'Proposal Comparisons', 
                                  timestamp, 'proposal_comparison',
                                  zoom_factor=0.30)
        plt.close()

    # 4. Voting Distribution Graph
    if not size_df.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_facecolor('none')
        fig.patch.set_alpha(0.0)
        
        unique_groups = sorted(size_df['power_group'].unique())
        x = np.arange(len(unique_groups))
        bottom = {group: 0 for group in unique_groups}
        colors = plt.cm.tab20(np.linspace(0, 1, len(vote_choices)))
        
        for choice, color in zip(vote_choices.values(), colors):
            choice_data = size_df[size_df['vote_choice'] == choice]
            
            heights = []
            for group in unique_groups:
                value = choice_data[choice_data['power_group'] == group]['total_voting_power']
                heights.append(float(value.iloc[0]) if len(value) > 0 else 0)
            
            bottom_values = [bottom[group] for group in unique_groups]
            ax.bar(x, heights, label=choice, bottom=bottom_values, color=color)
            
            for i, group in enumerate(unique_groups):
                bottom[group] += heights[i]
        
        ax.set_title('Voting Power Distribution by Size Category', pad=20, fontsize=12, color='black')
        ax.set_xlabel('Size Category', fontsize=10, color='black')
        ax.set_ylabel('Voting Power', fontsize=10, color='black')
        ax.set_xticks(x)
        ax.set_xticklabels(unique_groups, rotation=45, ha='right', color='black')
        ax.tick_params(colors='black')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.yaxis.set_major_formatter(FuncFormatter(millions_formatter))
        
        plt.tight_layout()
        create_plot_with_background(fig, ax, 'Voting Distribution', 
                                  timestamp, 'voting_distribution',
                                  zoom_factor=0.34)
        plt.close()

    total_current_power = 0
    total_current_voters = 0

    print("\nCurrent Proposal Results:")
    print("-" * 50)
    
    for choice in vote_choices.values():
        choice_data = lifecycle_df[lifecycle_df['vote_choice'] == choice]
        if not choice_data.empty:
            power = float(choice_data['total_power'].iloc[0])
            voters = int(choice_data['voters'].iloc[0])
            total_current_power += power
            total_current_voters += voters
            print(f"{choice}:")
            print(f"  Total Voting Power: {power:,.2f}")
            print(f"  Total Voters: {voters:,}")
    
    if not lifecycle_df.empty:
        winner = lifecycle_df.loc[lifecycle_df['total_power'].idxmax()]
        print(f"\nCurrently Winning: {winner['vote_choice']}")
        other_max = lifecycle_df[lifecycle_df['vote_choice'] != winner['vote_choice']]['total_power'].max()
        print(f"Leading by: {float(winner['total_power']) - float(other_max):,.2f} voting power")
        
        first_row = lifecycle_df.iloc[0]
        print(f"\nComparison against other proposals:")
        print(f"(Compared against {int(first_row['total_props'])} other proposals)")
        
        print(f"\nVoting Power:")
        print(f"- Current: {total_current_power:,.2f}")
        print(f"- Rank: #{int(first_row['power_rank'])} of {int(first_row['total_props']) + 1}")
        print(f"- Average: {float(first_row['avg_power']):,.2f} ({total_current_power/float(first_row['avg_power']):,.1f}x average)")
        print(f"- Median: {float(first_row['median_power']):,.2f} ({total_current_power/float(first_row['median_power']):,.1f}x median)")
        print(f"- Range: {float(first_row['min_power']):,.2f} to {float(first_row['max_power']):,.2f}")
        
        print(f"\nVoter Count:")
        print(f"- Current: {total_current_voters:,}")
        print(f"- Rank: #{int(first_row['voter_rank'])} of {int(first_row['total_props']) + 1}")
        print(f"- Average: {float(first_row['avg_voters']):,.0f} ({total_current_voters/float(first_row['avg_voters']):,.1f}x average)")
        print(f"- Median: {float(first_row['median_voters']):,.0f} ({total_current_voters/float(first_row['median_voters']):,.1f}x median)")
        print(f"- Range: {int(first_row['min_voters']):,} to {int(first_row['max_voters']):,}")
    
    if not size_df.empty:
        print("\nVoting Power Distribution Analysis:")
        print("-" * 50)
        
        for choice in vote_choices.values():
            choice_data = size_df[size_df['vote_choice'] == choice]
            if not choice_data.empty:
                total_power = float(choice_data.iloc[0]['total_power'])
                total_voters = int(choice_data.iloc[0]['total_voters'])
                
                print(f"\n{choice} - Total Power: {format_number(total_power)} ({total_voters:,} voters)")
                print("Size Category      Wallets    Power      % of Vote")
                print("-" * 60)
                
                for _, row in choice_data.iterrows():
                    power = float(row['total_voting_power'])
                    wallets = int(row['num_wallets'])
                    percentage = (power / total_power) * 100
                    print(f"{row['power_group']:<16} {wallets:>8,} {format_number(power):>10} {percentage:>8.1f}%")

        print("\nVoter Concentration Analysis:")
        for choice in vote_choices.values():
            choice_data = size_df[size_df['vote_choice'] == choice]
            if not choice_data.empty:
                large_holders = choice_data[choice_data['power_group'].str.contains('f/|g/|h/')]
                total_power = float(choice_data.iloc[0]['total_power'])
                whale_power = large_holders['total_voting_power'].sum()
                whale_wallets = large_holders['num_wallets'].sum()
                
                print(f"\n{choice}:")
                print(f"Top holders (>100K): {whale_wallets:,} wallets controlling {(whale_power/total_power)*100:.1f}% of vote")
                
                retail = choice_data[choice_data['power_group'].str.contains('a/|b/|c/')]
                retail_power = retail['total_voting_power'].sum()
                retail_wallets = retail['num_wallets'].sum()
                print(f"Retail (<1K): {retail_wallets:,} wallets controlling {(retail_power/total_power)*100:.1f}% of vote")
    
    return lifecycle_df, size_df, hourly_df

def generate_thread_analysis(lifecycle_df, size_df, hourly_df, prop_context, anthropic_key, timestamp):
    """
    Generate a Twitter thread analysis using voting data, graphs, and Claude API.
    """
    # Initialize Anthropic client
    anthropic = Anthropic(api_key=anthropic_key)
    
    # Format the data metrics
    total_power = lifecycle_df['total_power'].sum()
    total_voters = lifecycle_df['voters'].sum()
    
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    charts_dir = os.path.join(script_dir, 'charts')
    
    # Load the generated graphs as base64
    graph_paths = {
        'voting_power': os.path.join(charts_dir, f'voting_power_trend_{timestamp}.png'),
        'voters': os.path.join(charts_dir, f'voter_trend_{timestamp}.png'),
        'comparison': os.path.join(charts_dir, f'proposal_comparison_{timestamp}.png'),
        'distribution': os.path.join(charts_dir, f'voting_distribution_{timestamp}.png')
    }
    
    def encode_image(image_path):
        if os.path.exists(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        else:
            print(f"Warning: Could not find image at {image_path}")
            return None

    # Create initial message
    messages = [{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": """You'll analyze voting data and 4 graphs for a proposal. Your analysis should use the data shown in these visuals, but don't reference them explicitly. Write as if readers will see your tweets alongside the visuals."""
            },
            {
                "type": "text",
                "text": f"""PROPOSAL CONTEXT:
                {prop_context}

                KEY METRICS:
                - Total Voting Power: {total_power:,.1f}
                - Total Voters: {total_voters:,}"""
            }
        ]
    }]

    # Add each graph with verification
    for name, path in graph_paths.items():
        if os.path.exists(path):  # Verify file exists
            image_data = encode_image(path)
            if image_data:  # Only add if we successfully encoded the image
                messages[0]["content"].extend([
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_data
                        }
                    }
                ])
        else:
            print(f"Warning: Missing {name} graph at {path}")

    # Add the thread requirements
    messages[0]["content"].append({
        "type": "text",
        "text": """
        Create a 6-tweet thread analyzing this proposal and its voting data.

        TWEET FORMAT:
        - Tweet 1: Introduce the proposal (funding Jupuary for next 2 years, 700M JUP each year, needs 70% yes). End with "let's dive into the cast votes so far"
        
        - Tweet 2: Describe the voting power trends over time. Focus on key moments and shifts in momentum.
        
        - Tweet 3: Discuss how the number of voters has grown and any notable participation patterns.
        
        - Tweet 4: Compare this vote's engagement to other Jupiter proposals.
        
        - Tweet 5: Analyze how different wallet sizes are participating in the vote.
        
        - Tweet 6: End with "If you want to learn more about this @JupiterExchange vote, check out our @flipsidecrypto dashboard"

        STYLE RULES:
        - Start tweets with just the number (1/, 2/, etc.)
        - Tag @JupiterExchange when mentioning Jupiter
        - No emojis
        - Keep under 280 chars per tweet
        - Focus on patterns and trends, minimize specific numbers
        - Be direct and analytical
        - Don't mention graphs - write as if visuals are attached
        - Match Pine Analytics' concise, data-focused style
        """
    })

    # Get Claude's analysis
    try:
        response = anthropic.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            temperature=0.7,
            system="You are Pine Analytics, known for precise, data-driven analysis of blockchain governance. Your style is concise, analytical, and focuses on patterns rather than listing many numbers.",
            messages=messages
        )
        
        # Extract the text content from the response
        thread_content = response.content[0].text if isinstance(response.content, list) else response.content
        
        # Save the thread
        thread_path = os.path.join(charts_dir, f'twitter_thread_{timestamp}.txt')
        with open(thread_path, 'w') as f:
            f.write(thread_content)
        
        return thread_content, thread_path
        
    except Exception as e:
        print(f"Error generating thread: {str(e)}")
        return None, None


def main():
    # Make sure charts directory exists
    if not os.path.exists('charts'):
        os.makedirs('charts')
    
    api_key = input("Enter your Flipside API key: ")
    prop_id = input("Enter the proposal ID: ")
    
    vote_choices = {}
    choice_count = int(input("How many voting choices are there (max 6)? "))
    
    for i in range(1, min(choice_count + 1, 7)):
        choice_name = input(f"Enter the name for choice {i}: ").lower()
        vote_choices[i] = choice_name
    
    lifecycle_df, size_df, hourly_df = get_voting_analysis(api_key, prop_id, vote_choices)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save = input("\nWould you like to save the data to CSV? (y/n): ")
    if save.lower() == 'y':
        lifecycle_df.to_csv(f"charts/voting_lifecycle_{timestamp}.csv", index=False)
        size_df.to_csv(f"charts/voting_distribution_{timestamp}.csv", index=False)
        hourly_df.to_csv(f"charts/voting_hourly_{timestamp}.csv", index=False)
        print(f"Results saved to CSV files in charts folder with timestamp {timestamp}")
    
    print("\nGraphs have been saved as PNG files in the charts folder:")
    print(f"- voting_power_trend_{timestamp}.png")
    print(f"- voter_trend_{timestamp}.png")
    print(f"- proposal_comparison_{timestamp}.png")
    print(f"- voting_distribution_{timestamp}.png")

    # Add Claude analysis option
    generate_thread = input("\nWould you like to generate a Twitter thread analysis using Claude? (y/n): ")
    if generate_thread.lower() == 'y':
        anthropic_key = input("Enter your Anthropic API key: ").strip()
        if anthropic_key:
            print("\nPlease provide context about the proposal.")
            print("Include information such as:")
            print("- What is being voted on")
            print("- Why it's important")
            print("- Any relevant background information")
            print("- Timeline or deadlines")
            prop_context = input("\nProposal context: ").strip()
            
            try:
                thread, thread_path = generate_thread_analysis(lifecycle_df, size_df, hourly_df, 
                                                            prop_context, anthropic_key, timestamp)
                if thread:
                    print("\nGenerated Thread:")
                    print("-" * 50)
                    print(thread)
                    print(f"\nThread saved to {thread_path}")
                
            except Exception as e:
                print(f"\nError generating thread: {str(e)}")
                print("Please check your Anthropic API key and try again.")

if __name__ == "__main__":
    main()

#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

