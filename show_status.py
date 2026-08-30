import pandas as pd
import glob
import os

summary_files = glob.glob('cdt_logs/**/win_summary*.csv', recursive=True)

if not summary_files:
    print("\nלא נמצאו קובצי win_summary עדיין.")
else:
    df = pd.read_csv(summary_files[0])
    
    print("\n" + "="*50)
    print("      תקציר ביצועי המודל (Executive Summary)")
    print("="*50)
    
    total_wins = df['wins'].sum()
    total_losses = df['losses'].sum()
    total_ties = df['ties'].sum()
    total_windows = df['windows'].sum()
    
    win_pct = (total_wins / total_windows) * 100 if total_windows > 0 else 0
    
    print(f"סה\"כ חלונות שנבדקו: {total_windows}")
    print(f"ניצחונות למודל (MAB): {total_wins} ({win_pct:.1f}%)")
    print(f"הפסדים לבייסליין:     {total_losses}")
    print(f"תיקו:                   {total_ties}")
    print("-" * 50)
    
    print("\nחלוקה לפי קובצי עקבות (Traces):")
    for idx, row in df.iterrows():
        status = "🟢 מנצח" if row['wins'] > row['losses'] else ("🔴 מפסיד" if row['losses'] > row['wins'] else "⚪ תיקו")
        print(f"• {row['trace']:<15} | מטמון: {row['cache_size']:<5} | ניצחונות: {row['wins']}/{row['windows']} | סטטוס: {status}")
    print("="*50 + "\n")
