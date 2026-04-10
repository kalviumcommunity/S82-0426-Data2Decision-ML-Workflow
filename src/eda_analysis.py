import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from src.data_loader import load_data
from src.config import DATA_PATH, TEXT_FEATURES, TARGET_COLUMN

# Use non-interactive backend for server environments
import matplotlib
matplotlib.use('Agg')

def run_eda():
    print("\n" + "="*60)
    print("🔍 PROJECT DESCRIPTION ANALYZER: FEATURE DISTRIBUTION ANALYSIS")
    print("="*60)

    # 1. Load Dataset
    df = load_data(DATA_PATH)
    text_col = TEXT_FEATURES[0]

    # 2. Numerical Feature Analysis (adapted for NLP)
    print("\n📈 Deriving numerical metrics from text...")
    df['text_length'] = df[text_col].apply(len)
    df['word_count'] = df[text_col].apply(lambda x: len(str(x).split()))

    print("\n🔹 Descriptive Statistics:")
    print(df[['text_length', 'word_count']].describe())

    print(f"\n🔹 Skewness:")
    print(f"   - Text Length: {df['text_length'].skew():.4f}")
    print(f"   - Word Count:  {df['word_count'].skew():.4f}")

    # Create reports directory if not exists
    os.makedirs('reports', exist_ok=True)

    # Visualization: Word Count Distribution
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.histplot(df['word_count'], kde=True, color='skyblue')
    plt.title('Distribution of Word Counts')
    plt.xlabel('Word Count')
    
    plt.subplot(1, 2, 2)
    sns.boxplot(x=df['word_count'], color='lightgreen')
    plt.title('Boxplot of Word Counts')
    
    plt.tight_layout()
    plt.savefig('reports/word_count_distribution.png')
    print("✅ Saved: reports/word_count_distribution.png")

    # 3. Text Feature Analysis
    print("\n🔹 Top 10 Frequent Words (Excluding Stopwords):")
    cv = CountVectorizer(stop_words='english')
    words = cv.fit_transform(df[text_col])
    sum_words = words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in cv.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    
    for word, freq in words_freq[:10]:
        print(f"   - {word}: {freq}")

    # 4. Target Distribution
    print("\n🎯 Target Analysis (Multi-label Skills):")
    skill_counts = df[TARGET_COLUMN].sum().sort_values(ascending=False)
    print(skill_counts)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=skill_counts.values, y=skill_counts.index, palette='viridis')
    plt.title('Skill Distribution in Dataset')
    plt.xlabel('Frequency')
    plt.ylabel('Skill')
    plt.savefig('reports/skill_distribution.png')
    print("✅ Saved: reports/skill_distribution.png")

    # 5. Target-Based Comparison
    print("\n⚖️  Target-Based Comparison...")
    
    # For multi-label, we'll check if word count varies significantly when a skill is present vs absent
    # We'll pick the most frequent skill for simplified visualization
    top_skill = skill_counts.index[0]
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.boxplot(x=df[top_skill], y=df['word_count'], palette='Pastel1')
    plt.title(f'Word Count vs Presence of {top_skill}')
    plt.xlabel(f'Presence of {top_skill} (0=No, 1=Yes)')
    
    plt.subplot(1, 2, 2)
    sns.boxplot(x=df[top_skill], y=df['text_length'], palette='Pastel2')
    plt.title(f'Text Length vs Presence of {top_skill}')
    plt.xlabel(f'Presence of {top_skill} (0=No, 1=Yes)')
    
    plt.tight_layout()
    plt.savefig('reports/target_comparison.png')
    print("✅ Saved: reports/target_comparison.png")

    print("\n" + "="*60)
    print("🏁 EDA COMPLETE")
    print("="*60)

if __name__ == "__main__":
    import sys
    import os
    # Ensure project root is in path
    sys.path.append(os.getcwd())
    run_eda()
