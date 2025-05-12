import os
import numpy as np
import pandas as pd
import argparse
from utils.skeleton_gait import GaitFeatureExtractor
from utils.gait_validator import GaitValidator
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Test Gait Analysis System")
    parser.add_argument("--features", type=str, default="industrial_gait_features.csv", 
                       help="Path to saved gait features CSV file")
    parser.add_argument("--identities", type=str, default="person_identities.json",
                       help="Path to identities JSON file (optional)")
    parser.add_argument("--output", type=str, default="gait_validation_results",
                       help="Directory to save validation results")
    parser.add_argument("--test_size", type=float, default=0.3,
                       help="Proportion of data to use for testing (default: 0.3)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    print("=== Gait Analysis System Validation ===")
    
    # Initialize validator
    validator = GaitValidator()
    
    # Load feature data
    print("\n1. Loading feature data...")
    if not validator.load_features_from_csv(args.features):
        print("Error loading features. Exiting.")
        return
    
    # Load identities if available
    if os.path.exists(args.identities):
        print("\n2. Loading identities...")
        validator.load_identities_from_json(args.identities)
    else:
        print("\n2. No identities file found. Using track IDs as identities.")
    
    # Analyze feature quality
    print("\n3. Analyzing feature quality...")
    quality_results = validator.analyze_feature_quality()
    
    if 'feature_quality' in quality_results:
        print("\nTop 5 most discriminative features:")
        top_features = list(quality_results['feature_quality'].items())[:5]
        for feature, score in top_features:
            print(f"- {feature}: {score:.4f}")
            
        # Visualize feature importance
        print("\n4. Visualizing feature importance...")
        validator.visualize_feature_importance(
            top_n=10, 
            save_path=os.path.join(args.output, "feature_importance.png")
        )
    
    # Train classifier
    print("\n5. Training classifier...")
    validator.train_classifier()
    
    # Evaluate classifier
    print("\n6. Evaluating classifier...")
    results = validator.evaluate_classifier()
    
    if results:
        print("\n7. Visualizing confusion matrix...")
        validator.visualize_confusion_matrix(
            save_path=os.path.join(args.output, "confusion_matrix.png")
        )
        
        # Save validation results summary
        summary_path = os.path.join(args.output, "validation_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("=== Gait Analysis System Validation Summary ===\n\n")
            f.write(f"Feature file: {args.features}\n")
            f.write(f"Total samples: {len(validator.features_df)}\n")
            if 'missing_values' in quality_results:
                avg_missing = np.mean(list(quality_results['missing_values']['percentages'].values()))
                f.write(f"Average missing values: {avg_missing:.2f}%\n\n")
            f.write("Performance Metrics:\n")
            f.write(f"- Accuracy: {results['accuracy']:.4f}\n")
            f.write(f"- Precision: {results['precision']:.4f}\n")
            f.write(f"- Recall: {results['recall']:.4f}\n")
            f.write(f"- F1 Score: {results['f1_score']:.4f}\n\n")
            
            if 'feature_quality' in quality_results:
                f.write("Top 10 Most Discriminative Features:\n")
                top_features = list(quality_results['feature_quality'].items())[:10]
                for i, (feature, score) in enumerate(top_features, 1):
                    f.write(f"{i}. {feature}: {score:.4f}\n")
                    
        print(f"\nValidation summary saved to {summary_path}")
        
        # Save model
        model_path = os.path.join(args.output, "gait_classifier_model.pkl")
        validator.save_model(model_path)

if __name__ == "__main__":
    main()