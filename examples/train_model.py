#!/usr/bin/env python3
"""
Complete training pipeline for customer churn prediction models.

This script demonstrates how to use the modular components to:
1. Load and preprocess data
2. Train multiple models
3. Perform hyperparameter tuning
4. Evaluate models comprehensively
5. Save the best models

Usage:
    python examples/train_model.py
"""

import sys
import os
import time
from datetime import datetime

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_preprocessing import preprocess_pipeline
from src.model_training import ModelTrainer
from src.evaluation import evaluate_models_comprehensive
from src.utils import (
    setup_plotting_style, create_churn_distribution_plot,
    create_numerical_histograms, create_categorical_churn_analysis,
    create_correlation_heatmap, display_correlation_with_target,
    calculate_business_impact, print_section_header, format_duration
)
from src.config import RANDOM_STATE


def run_complete_pipeline(perform_tuning: bool = True, save_models: bool = True) -> None:
    """
    Run the complete machine learning pipeline.
    
    Args:
        perform_tuning (bool): Whether to perform hyperparameter tuning
        save_models (bool): Whether to save trained models
    """
    start_time = time.time()
    
    print_section_header("CUSTOMER CHURN PREDICTION - TRAINING PIPELINE")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Random state: {RANDOM_STATE}")
    print()
    
    # Set up plotting style
    setup_plotting_style()
    
    try:
        # ========================
        # PHASE 1: DATA PREPROCESSING
        # ========================
        print_section_header("PHASE 1: DATA PREPROCESSING")
        
        phase_start = time.time()
        X_train, X_test, y_train, y_test, preprocessor = preprocess_pipeline()
        
        print(f"✓ Data preprocessing completed in {format_duration(time.time() - phase_start)}")
        
        # ========================
        # PHASE 2: EXPLORATORY DATA ANALYSIS
        # ========================
        print_section_header("PHASE 2: EXPLORATORY DATA ANALYSIS")
        
        # Load original data for visualization
        from src.data_preprocessing import load_data, clean_data
        df = load_data()
        df_clean = clean_data(df)
        
        print("Creating visualizations...")
        create_churn_distribution_plot(df_clean)
        create_numerical_histograms(df_clean)
        create_categorical_churn_analysis(df_clean)
        create_correlation_heatmap(df_clean)
        
        # Display correlation analysis
        correlation_df = display_correlation_with_target(df_clean)
        
        # ========================
        # PHASE 3: MODEL TRAINING
        # ========================
        print_section_header("PHASE 3: MODEL TRAINING")
        
        phase_start = time.time()
        
        # Initialize trainer
        trainer = ModelTrainer(preprocessor)
        trainer.initialize_models(include_neural_network=True)
        
        print(f"Initialized {len(trainer.models)} models:")
        for name in trainer.models.keys():
            print(f"  • {name}")
        
        # Train baseline models
        print("\nTraining baseline models...")
        trained_models = trainer.train_models(X_train, y_train)
        
        # Perform cross-validation
        print("\nPerforming cross-validation...")
        cv_scores = trainer.perform_cross_validation(X_train, y_train, scoring='roc_auc')
        
        print(f"✓ Model training completed in {format_duration(time.time() - phase_start)}")
        
        # ========================
        # PHASE 4: HYPERPARAMETER TUNING (Optional)
        # ========================
        if perform_tuning:
            print_section_header("PHASE 4: HYPERPARAMETER TUNING")
            
            phase_start = time.time()
            
            # Select top models for tuning based on cross-validation
            top_models = sorted(cv_scores.items(), key=lambda x: x[1]['mean'], reverse=True)[:3]
            top_model_names = [name for name, _ in top_models]
            
            print(f"Tuning top {len(top_model_names)} models: {top_model_names}")
            
            tuning_results = trainer.hyperparameter_tuning(
                X_train, y_train, 
                model_names=top_model_names,
                scoring='roc_auc'
            )
            
            print(f"✓ Hyperparameter tuning completed in {format_duration(time.time() - phase_start)}")
            
            # Use best models for evaluation
            models_to_evaluate = {**trained_models, **trainer.best_models}
        else:
            models_to_evaluate = trained_models
        
        # ========================
        # PHASE 5: MODEL EVALUATION
        # ========================
        print_section_header("PHASE 5: MODEL EVALUATION")
        
        phase_start = time.time()
        
        evaluator, comparison_df = evaluate_models_comprehensive(
            models_to_evaluate, X_test, y_test, show_plots=True
        )
        
        # Generate detailed reports for top models
        print("\nDetailed Model Reports:")
        print("=" * 60)
        
        best_models = comparison_df.head(3)['Model'].tolist()
        for model_name in best_models:
            print(evaluator.generate_model_report(model_name))
        
        # Feature importance analysis
        print_section_header("FEATURE IMPORTANCE ANALYSIS")
        
        for model_name in best_models:
            importance_df = trainer.get_feature_importance(model_name)
            if importance_df is not None:
                print(f"\nTop 10 Important Features for {model_name}:")
                print(importance_df.head(10).to_string(index=False))
                
                # Plot feature importance
                from src.evaluation import ModelEvaluator
                eval_temp = ModelEvaluator()
                eval_temp.plot_feature_importance(importance_df, model_name, top_n=15)
        
        # Business impact analysis
        print_section_header("BUSINESS IMPACT ANALYSIS")
        
        business_impact = calculate_business_impact(
            evaluator.evaluation_results,
            customer_value=1000,  # $1000 average customer value
            retention_cost=100    # $100 retention campaign cost
        )
        
        print("Business Impact Analysis:")
        print(business_impact.to_string(index=False))
        
        print(f"✓ Model evaluation completed in {format_duration(time.time() - phase_start)}")
        
        # ========================
        # PHASE 6: MODEL PERSISTENCE
        # ========================
        if save_models:
            print_section_header("PHASE 6: MODEL PERSISTENCE")
            
            phase_start = time.time()
            
            # Save top 3 models
            top_models_to_save = best_models
            saved_paths = trainer.save_models(top_models_to_save)
            
            print("Saved models:")
            for model_name, path in saved_paths.items():
                print(f"  • {model_name}: {path}")
            
            print(f"✓ Model persistence completed in {format_duration(time.time() - phase_start)}")
        
        # ========================
        # FINAL SUMMARY
        # ========================
        print_section_header("PIPELINE SUMMARY")
        
        total_duration = time.time() - start_time
        
        print(f"Total execution time: {format_duration(total_duration)}")
        print(f"Models trained: {len(trained_models)}")
        
        if perform_tuning:
            print(f"Models tuned: {len(trainer.best_models)}")
        
        best_model = evaluator.get_best_model('roc_auc')
        if best_model:
            best_score = evaluator.evaluation_results[best_model]['metrics']['roc_auc']
            print(f"Best model: {best_model} (ROC AUC: {best_score:.4f})")
        
        print("\n🎉 Training pipeline completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error in training pipeline: {e}")
        raise


def main():
    """Main function for script execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train customer churn prediction models')
    parser.add_argument('--no-tuning', action='store_true',
                       help='Skip hyperparameter tuning')
    parser.add_argument('--no-save', action='store_true',
                       help='Skip saving models')
    parser.add_argument('--quick', action='store_true',
                       help='Quick run without tuning and saving')
    
    args = parser.parse_args()
    
    perform_tuning = not (args.no_tuning or args.quick)
    save_models = not (args.no_save or args.quick)
    
    run_complete_pipeline(perform_tuning=perform_tuning, save_models=save_models)


if __name__ == "__main__":
    main()