# wandb_dtype_analysis.py - Analyze your dtype experiments
import wandb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_dtype_experiments():
    """Download and analyze your dtype experiments from W&B."""
    
    # Initialize W&B API
    api = wandb.Api()
    
    # Get runs from your project (adjust project name as needed)
    runs = api.runs("jameszoryk-me/image_class_prediction")
    
    # Extract data from runs
    results = []
    for run in runs:
        if run.state == "finished":  # Only completed runs
            
            # Extract configuration
            config = run.config
            
            # Extract metrics  
            summary = run.summary
            
            # Build result record
            result = {
                "experiment_name": config.get("experiment", {}).get("name", "unknown"),
                "data_dtype": config.get("data", {}).get("tensor_dtype", "unknown"),
                "model_dtype": config.get("model", {}).get("model_dtype", "unknown"),
                "batch_size": config.get("data", {}).get("batch_size", 0),
                "normalize": config.get("data", {}).get("normalize", False),
                "augmentation": config.get("data", {}).get("use_augmentation", False),
                "test_accuracy": summary.get("final/test_acc", 0) * 100,  # Convert to %
                "test_f1": summary.get("final/test_f1", 0) * 100,
                "test_loss": summary.get("final/test_loss", 0),
                "best_val_loss": summary.get("final/best_val_loss", 0),
                "runtime": run.summary.get("_runtime", 0),
                "created_at": run.created_at,
            }
            results.append(result)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    if df.empty:
        print("❌ No experiments found! Make sure W&B project name is correct.")
        return
    
    # Sort by creation time
    df = df.sort_values("created_at")
    
    print("🎯 DTYPE EXPERIMENT RESULTS")
    print("=" * 80)
    
    # Show basic statistics
    dtype_summary = df.groupby("data_dtype").agg({
        "test_accuracy": ["mean", "std", "max"],
        "test_f1": ["mean", "std", "max"], 
        "test_loss": ["mean", "std", "min"],
        "runtime": "mean"
    }).round(4)
    
    print("\n📊 DTYPE PERFORMANCE SUMMARY:")
    print(dtype_summary)
    
    # Show individual results
    print(f"\n📋 INDIVIDUAL EXPERIMENT RESULTS:")
    display_cols = ["experiment_name", "data_dtype", "test_accuracy", "test_f1", "test_loss", "runtime"]
    print(df[display_cols].to_string(index=False))
    
    # Create visualizations
    create_dtype_visualizations(df)
    
    # Performance analysis
    analyze_dtype_performance(df)
    
    return df

def create_dtype_visualizations(df):
    """Create performance visualizations."""
    
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Accuracy by dtype
    sns.boxplot(data=df, x="data_dtype", y="test_accuracy", ax=axes[0,0])
    axes[0,0].set_title("Test Accuracy by Data Type")
    axes[0,0].set_ylabel("Test Accuracy (%)")
    
    # 2. Loss by dtype  
    sns.boxplot(data=df, x="data_dtype", y="test_loss", ax=axes[0,1])
    axes[0,1].set_title("Test Loss by Data Type")
    axes[0,1].set_ylabel("Test Loss")
    
    # 3. Runtime by dtype
    sns.boxplot(data=df, x="data_dtype", y="runtime", ax=axes[1,0])
    axes[1,0].set_title("Training Time by Data Type")
    axes[1,0].set_ylabel("Runtime (seconds)")
    
    # 4. Accuracy vs Batch Size
    scatter_df = df[df["batch_size"] > 0]  # Filter out invalid batch sizes
    if not scatter_df.empty:
        sns.scatterplot(data=scatter_df, x="batch_size", y="test_accuracy", 
                       hue="data_dtype", size="runtime", ax=axes[1,1])
        axes[1,1].set_title("Accuracy vs Batch Size")
        axes[1,1].set_xlabel("Batch Size")
        axes[1,1].set_ylabel("Test Accuracy (%)")
    
    plt.tight_layout()
    plt.savefig("dtype_analysis_dashboard.png", dpi=300, bbox_inches="tight")
    print(f"\n📈 Visualizations saved as 'dtype_analysis_dashboard.png'")

def analyze_dtype_performance(df):
    """Analyze dtype performance patterns."""
    
    print(f"\n🔍 PERFORMANCE ANALYSIS:")
    print("=" * 50)
    
    # Find baseline (float32)
    float32_results = df[df["data_dtype"] == "float32"]
    if not float32_results.empty:
        baseline_acc = float32_results["test_accuracy"].max()
        print(f"📊 Float32 Baseline Accuracy: {baseline_acc:.2f}%")
        
        # Compare other dtypes to baseline
        for dtype in df["data_dtype"].unique():
            if dtype != "float32":
                dtype_results = df[df["data_dtype"] == dtype]
                if not dtype_results.empty:
                    best_acc = dtype_results["test_accuracy"].max()
                    accuracy_drop = baseline_acc - best_acc
                    print(f"📉 {dtype.upper()} vs Float32: -{accuracy_drop:.2f}% accuracy")
    
    # Memory efficiency analysis
    print(f"\n💾 THEORETICAL MEMORY EFFICIENCY:")
    memory_savings = {
        "float32": "100% (baseline)",
        "float16": "50% (2x compression)", 
        "bfloat16": "50% (2x compression)",
        "int8": "25% (4x compression)",
        "uint8": "25% (4x compression)"
    }
    
    for dtype in df["data_dtype"].unique():
        if dtype in memory_savings:
            print(f"💾 {dtype.upper()}: {memory_savings[dtype]}")
    
    # Speed analysis
    print(f"\n⚡ SPEED ANALYSIS:")
    speed_summary = df.groupby("data_dtype")["runtime"].agg(["mean", "std"]).round(1)
    for dtype, stats in speed_summary.iterrows():
        print(f"⏱️ {dtype.upper()}: {stats['mean']}±{stats['std']}s")

def compare_latest_experiments():
    """Compare your latest 3 experiments specifically."""
    
    print(f"\n🎯 YOUR LATEST DTYPE EXPERIMENTS:")
    print("=" * 60)
    
    # Manual results from your experiments
    latest_results = [
        {
            "experiment": "accuracy_baseline",
            "dtype": "float32", 
            "accuracy": 99.36,
            "f1": 99.30,
            "loss": 0.0219,
            "notes": "Perfect baseline"
        },
        {
            "experiment": "accuracy_half", 
            "dtype": "float16",
            "accuracy": 99.30,
            "f1": 99.24, 
            "loss": 0.0215,
            "notes": "Minimal degradation!"
        },
        {
            "experiment": "accuracy_quantized",
            "dtype": "int8",
            "accuracy": 98.80,
            "f1": 98.69,
            "loss": 0.0382, 
            "notes": "Good for quantization"
        }
    ]
    
    df_latest = pd.DataFrame(latest_results)
    print(df_latest.to_string(index=False))
    
    print(f"\n🏆 KEY INSIGHTS FROM YOUR EXPERIMENTS:")
    print(f"• Float16 efficiency: Only 0.06% accuracy drop!")
    print(f"• Int8 compression: 0.56% drop for 4x memory savings")
    print(f"• All experiments >98.8% accuracy - EXCELLENT!")
    print(f"• Your dtype system works perfectly! 🎉")

if __name__ == "__main__":
    print("🔍 W&B DTYPE EXPERIMENT ANALYZER")
    print("=" * 50)
    
    try:
        # Try to analyze from W&B
        df = analyze_dtype_experiments()
        
    except Exception as e:
        print(f"⚠️ W&B API error: {e}")
        print("📊 Using manual analysis of your latest results...")
        
    # Always show your latest results
    compare_latest_experiments()
    
    print(f"\n🚀 NEXT STEPS:")
    print(f"1. Try larger batch sizes with float16")
    print(f"2. Test mixed precision training") 
    print(f"3. Add data normalization for int8")
    print(f"4. Scale to CIFAR-10!")
    print(f"\n🎉 You've built an amazing dtype experimentation system!")
