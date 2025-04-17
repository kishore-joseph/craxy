import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
import logging
import argparse
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("aml_model_training.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("AML_MODEL_TRAINER")

class AMLModelTrainer:
    """
    Trains and evaluates machine learning models for AML transaction monitoring.
    """
    
    def __init__(self, data_dir, output_dir=None):
        """
        Initialize the model trainer.
        
        Args:
            data_dir: Directory containing the prepared ML data
            output_dir: Directory to save model outputs
        """
        self.data_dir = data_dir
        self.output_dir = output_dir or os.path.join(data_dir, 'models')
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.rule_targets_train = None
        self.rule_targets_test = None
        
        self.models = {}
        self.rule_models = {}
        self.best_model = None
        self.feature_importances = {}
    
    def load_data(self):
        """
        Load the prepared data from CSV files.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load feature and target data
            self.X_train = pd.read_csv(os.path.join(self.data_dir, 'X_train.csv'))
            self.X_test = pd.read_csv(os.path.join(self.data_dir, 'X_test.csv'))
            self.y_train = pd.read_csv(os.path.join(self.data_dir, 'y_train.csv')).values.ravel()
            self.y_test = pd.read_csv(os.path.join(self.data_dir, 'y_test.csv')).values.ravel()
            
            # Load rule-specific targets if available
            rule_train_path = os.path.join(self.data_dir, 'rule_targets_train.csv')
            rule_test_path = os.path.join(self.data_dir, 'rule_targets_test.csv')
            
            if os.path.exists(rule_train_path) and os.path.exists(rule_test_path):
                self.rule_targets_train = pd.read_csv(rule_train_path)
                self.rule_targets_test = pd.read_csv(rule_test_path)
                logger.info("Rule-specific target data loaded successfully")
            
            logger.info(f"Data loaded successfully. Training set: {self.X_train.shape}, Test set: {self.X_test.shape}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return False
    
    def exploratory_data_analysis(self):
        """
        Perform exploratory data analysis and save visualizations.
        """
        try:
            eda_dir = os.path.join(self.output_dir, 'eda')
            os.makedirs(eda_dir, exist_ok=True)
            
            # Basic statistics
            stats = self.X_train.describe().T
            stats.to_csv(os.path.join(eda_dir, 'feature_statistics.csv'))
            
            # Class distribution
            plt.figure(figsize=(8, 6))
            sns.countplot(y=self.y_train)
            plt.title('Class Distribution in Training Data')
            plt.xlabel('Count')
            plt.ylabel('Class')
            plt.savefig(os.path.join(eda_dir, 'class_distribution.png'))
            plt.close()
            
            # Rule distribution if available
            if self.rule_targets_train is not None:
                rule_counts = self.rule_targets_train.sum().sort_values(ascending=False)
                
                plt.figure(figsize=(12, 8))
                rule_counts.plot(kind='bar')
                plt.title('AML Rule Triggers Distribution')
                plt.xlabel('Rule')
                plt.ylabel('Count')
                plt.xticks(rotation=90)
                plt.tight_layout()
                plt.savefig(os.path.join(eda_dir, 'rule_distribution.png'))
                plt.close()
            
            # Correlation matrix of features
            plt.figure(figsize=(20, 16))
            corr_matrix = self.X_train.corr()
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', annot=False, 
                        linewidths=0.5, vmin=-1, vmax=1)
            plt.title('Feature Correlation Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join(eda_dir, 'correlation_matrix.png'))
            plt.close()
            
            # Save top correlated features
            top_corr = corr_matrix.unstack().sort_values(ascending=False)
            top_corr = top_corr[top_corr < 1]  # Remove self-correlations
            top_corr = top_corr[abs(top_corr) > 0.5]  # Keep only strong correlations
            top_corr.to_csv(os.path.join(eda_dir, 'top_correlations.csv'))
            
            logger.info(f"Exploratory data analysis completed. Results saved to {eda_dir}")
            
        except Exception as e:
            logger.error(f"Error in exploratory data analysis: {str(e)}")
    
    def train_models(self, use_smote=True, n_jobs=-1):
        """
        Train multiple machine learning models for AML detection.
        
        Args:
            use_smote: Whether to use SMOTE for handling class imbalance
            n_jobs: Number of parallel jobs for training
            
        Returns:
            dict: Dictionary of trained models with evaluation metrics
        """
        try:
            logger.info("Starting model training")
            
            # Define models to train
            models_to_train = {
                'logistic_regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
                'random_forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
                'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
                'xgboost': xgb.XGBClassifier(n_estimators=100, random_state=42, scale_pos_weight=sum(self.y_train==0)/sum(self.y_train==1)),
                'lightgbm': lgb.LGBMClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
                'neural_network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
            }
            
            # Define preprocessing steps
            preprocessor = StandardScaler()
            
            # Create models with preprocessing
            for name, model in models_to_train.items():
                logger.info(f"Training {name} model")
                
                try:
                    if use_smote:
                        # Use SMOTE to handle class imbalance
                        pipeline = ImbPipeline([
                            ('preprocessor', preprocessor),
                            ('smote', SMOTE(random_state=42)),
                            ('classifier', model)
                        ])
                    else:
                        # Standard pipeline without SMOTE
                        pipeline = ImbPipeline([
                            ('preprocessor', preprocessor),
                            ('classifier', model)
                        ])
                    
                    # Train the model
                    pipeline.fit(self.X_train, self.y_train)
                    
                    # Evaluate the model
                    train_score = pipeline.score(self.X_train, self.y_train)
                    test_score = pipeline.score(self.X_test, self.y_test)
                    
                    # Make predictions
                    y_pred = pipeline.predict(self.X_test)
                    y_pred_proba = pipeline.predict_proba(self.X_test)[:, 1]
                    
                    # Calculate metrics
                    classification_rep = classification_report(self.y_test, y_pred, output_dict=True)
                    conf_matrix = confusion_matrix(self.y_test, y_pred)
                    
                    # Calculate ROC curve and AUC
                    fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
                    roc_auc = auc(fpr, tpr)
                    
                    # Calculate Precision-Recall curve
                    precision, recall, _ = precision_recall_curve(self.y_test, y_pred_proba)
                    pr_auc = auc(recall, precision)
                    
                    # Store model and metrics
                    self.models[name] = {
                        'pipeline': pipeline,
                        'metrics': {
                            'train_accuracy': train_score,
                            'test_accuracy': test_score,
                            'classification_report': classification_rep,
                            'confusion_matrix': conf_matrix.tolist(),
                            'roc_auc': roc_auc,
                            'pr_auc': pr_auc,
                            'fpr': fpr.tolist(),
                            'tpr': tpr.tolist(),
                            'precision': precision.tolist(),
                            'recall': recall.tolist()
                        }
                    }
                    
                    # Extract feature importance if available
                    if hasattr(model, 'feature_importances_') or (name == 'logistic_regression' and hasattr(model, 'coef_')):
                        self._extract_feature_importance(name, pipeline, self.X_train.columns)
                    
                    logger.info(f"{name} model trained successfully. Test accuracy: {test_score:.4f}, ROC AUC: {roc_auc:.4f}")
                    
                except Exception as e:
                    logger.error(f"Error training {name} model: {str(e)}")
                    continue
            
            # Determine the best model based on ROC AUC
            best_model_name = max(self.models, key=lambda name: self.models[name]['metrics']['roc_auc'])
            self.best_model = self.models[best_model_name]
            
            logger.info(f"Best model: {best_model_name} with ROC AUC: {self.best_model['metrics']['roc_auc']:.4f}")
            
            # Save visualizations and model results
            self._save_model_visualizations()
            self._save_model_results()
            
            return self.models
            
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            return {}
    
    def train_rule_specific_models(self, min_samples=50, use_smote=True):
        """
        Train models for specific AML rules with sufficient examples.
        
        Args:
            min_samples: Minimum number of positive examples required
            use_smote: Whether to use SMOTE for handling class imbalance
            
        Returns:
            dict: Dictionary of rule-specific models with evaluation metrics
        """
        if self.rule_targets_train is None or self.rule_targets_test is None:
            logger.warning("Rule-specific target data not available. Skipping rule-specific model training.")
            return {}
        
        try:
            logger.info("Starting rule-specific model training")
            
            # Find rules with sufficient positive examples
            rule_counts = self.rule_targets_train.sum()
            eligible_rules = rule_counts[rule_counts >= min_samples].index.tolist()
            
            if not eligible_rules:
                logger.warning(f"No rules have {min_samples} or more positive examples. Skipping rule-specific model training.")
                return {}
            
            logger.info(f"Training models for {len(eligible_rules)} rules with {min_samples}+ examples: {eligible_rules}")
            
            # Create directory for rule-specific models
            rule_model_dir = os.path.join(self.output_dir, 'rule_models')
            os.makedirs(rule_model_dir, exist_ok=True)
            
            for rule in eligible_rules:
                logger.info(f"Training model for rule {rule}")
                
                # Extract target data for this rule
                y_rule_train = self.rule_targets_train[rule].values
                y_rule_test = self.rule_targets_test[rule].values
                
                # Skip if no positive examples in test set
                if sum(y_rule_test) == 0:
                    logger.warning(f"No positive examples for rule {rule} in test set. Skipping.")
                    continue
                
                try:
                    # Use XGBoost for rule-specific models
                    model = xgb.XGBClassifier(
                        n_estimators=100,
                        max_depth=5,
                        learning_rate=0.1,
                        scale_pos_weight=sum(y_rule_train==0)/max(1, sum(y_rule_train==1)),
                        random_state=42
                    )
                    
                    # Create pipeline with preprocessing
                    preprocessor = StandardScaler()
                    
                    if use_smote and sum(y_rule_train) >= 10:  # Need at least 10 samples for SMOTE
                        pipeline = ImbPipeline([
                            ('preprocessor', preprocessor),
                            ('smote', SMOTE(random_state=42)),
                            ('classifier', model)
                        ])
                    else:
                        pipeline = ImbPipeline([
                            ('preprocessor', preprocessor),
                            ('classifier', model)
                        ])
                    
                    # Train the model
                    pipeline.fit(self.X_train, y_rule_train)
                    
                    # Evaluate the model
                    train_score = pipeline.score(self.X_train, y_rule_train)
                    test_score = pipeline.score(self.X_test, y_rule_test)
                    
                    # Make predictions
                    y_pred = pipeline.predict(self.X_test)
                    y_pred_proba = pipeline.predict_proba(self.X_test)[:, 1]
                    
                    # Calculate metrics
                    classification_rep = classification_report(y_rule_test, y_pred, output_dict=True)
                    conf_matrix = confusion_matrix(y_rule_test, y_pred)
                    
                    # Calculate ROC curve and AUC
                    fpr, tpr, _ = roc_curve(y_rule_test, y_pred_proba)
                    roc_auc = auc(fpr, tpr)
                    
                    # Calculate Precision-Recall curve
                    precision, recall, _ = precision_recall_curve(y_rule_test, y_pred_proba)
                    pr_auc = auc(recall, precision)
                    
                    # Store model and metrics
                    self.rule_models[rule] = {
                        'pipeline': pipeline,
                        'metrics': {
                            'train_accuracy': train_score,
                            'test_accuracy': test_score,
                            'classification_report': classification_rep,
                            'confusion_matrix': conf_matrix.tolist(),
                            'roc_auc': roc_auc,
                            'pr_auc': pr_auc,
                            'positive_samples_train': int(sum(y_rule_train)),
                            'positive_samples_test': int(sum(y_rule_test))
                        }
                    }
                    
                    # Extract feature importance
                    self._extract_feature_importance(f"rule_{rule}", pipeline, self.X_train.columns)
                    
                    logger.info(f"Model for rule {rule} trained successfully. Test accuracy: {test_score:.4f}, ROC AUC: {roc_auc:.4f}")
                    
                    # Save rule-specific visualizations
                    self._save_rule_model_visualizations(rule)
                    
                except Exception as e:
                    logger.error(f"Error training model for rule {rule}: {str(e)}")
                    continue
            
            # Save rule model results
            if self.rule_models:
                rule_results = {
                    rule: {
                        'metrics': {k: v for k, v in model_data['metrics'].items() 
                                   if k not in ['fpr', 'tpr', 'precision', 'recall']}
                    }
                    for rule, model_data in self.rule_models.items()
                }
                
                with open(os.path.join(rule_model_dir, 'rule_model_results.json'), 'w') as f:
                    json.dump(rule_results, f, indent=2)
            
            logger.info(f"Rule-specific model training completed. Trained {len(self.rule_models)} rule models.")
            
            return self.rule_models
            
        except Exception as e:
            logger.error(f"Error in rule-specific model training: {str(e)}")
            return {}
    
    def _extract_feature_importance(self, model_name, pipeline, feature_names):
        """Extract feature importance from the model if available."""
        try:
            classifier = pipeline.named_steps['classifier']
            
            if hasattr(classifier, 'feature_importances_'):
                importances = classifier.feature_importances_
            elif hasattr(classifier, 'coef_'):
                importances = np.abs(classifier.coef_[0])
            else:
                return
            
            # Create DataFrame of feature importances
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            })
            
            # Sort by importance
            feature_importance = feature_importance.sort_values('importance', ascending=False)
            
            # Store feature importance
            self.feature_importances[model_name] = feature_importance
            
        except Exception as e:
            logger.error(f"Error extracting feature importance for {model_name}: {str(e)}")
    
    def _save_model_visualizations(self):
        """Save visualizations for model performance."""
        try:
            # Create visualization directory
            viz_dir = os.path.join(self.output_dir, 'visualizations')
            os.makedirs(viz_dir, exist_ok=True)
            
            # ROC curve comparison
            plt.figure(figsize=(10, 8))
            for name, model_data in self.models.items():
                metrics = model_data['metrics']
                plt.plot(metrics['fpr'], metrics['tpr'], label=f"{name} (AUC = {metrics['roc_auc']:.3f})")
            
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve Comparison')
            plt.legend(loc='lower right')
            plt.savefig(os.path.join(viz_dir, 'roc_curve_comparison.png'))
            plt.close()
            
            # Precision-Recall curve comparison
            plt.figure(figsize=(10, 8))
            for name, model_data in self.models.items():
                metrics = model_data['metrics']
                plt.plot(metrics['recall'], metrics['precision'], label=f"{name} (AUC = {metrics['pr_auc']:.3f})")
            
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve Comparison')
            plt.legend(loc='lower left')
            plt.savefig(os.path.join(viz_dir, 'pr_curve_comparison.png'))
            plt.close()
            
            # Feature importance plots
            for model_name, importance_df in self.feature_importances.items():
                if model_name.startswith('rule_'):
                    continue  # Skip rule-specific models here
                    
                # Plot top 20 features
                top_features = importance_df.head(20)
                
                plt.figure(figsize=(12, 8))
                sns.barplot(x='importance', y='feature', data=top_features)
                plt.title(f'Top 20 Feature Importances - {model_name}')
                plt.tight_layout()
                plt.savefig(os.path.join(viz_dir, f'feature_importance_{model_name}.png'))
                plt.close()
            
            # Confusion matrix for best model
            best_model_name = max(self.models, key=lambda name: self.models[name]['metrics']['roc_auc'])
            conf_matrix = np.array(self.models[best_model_name]['metrics']['confusion_matrix'])
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {best_model_name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(os.path.join(viz_dir, 'confusion_matrix_best_model.png'))
            plt.close()
            
            logger.info(f"Model visualizations saved to {viz_dir}")
            
        except Exception as e:
            logger.error(f"Error saving model visualizations: {str(e)}")
    
    def _save_rule_model_visualizations(self, rule):
        """Save visualizations for a rule-specific model."""
        try:
            # Create visualization directory
            rule_viz_dir = os.path.join(self.output_dir, 'visualizations', 'rule_models')
            os.makedirs(rule_viz_dir, exist_ok=True)
            
            model_data = self.rule_models[rule]
            metrics = model_data['metrics']
            
            # ROC curve
            plt.figure(figsize=(8, 6))
            fpr, tpr, _ = roc_curve(self.rule_targets_test[rule].values, 
                                    model_data['pipeline'].predict_proba(self.X_test)[:, 1])
            plt.plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']:.3f}")
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - Rule {rule}')
            plt.legend(loc='lower right')
            plt.savefig(os.path.join(rule_viz_dir, f'roc_curve_{rule}.png'))
            plt.close()
            
            # Precision-Recall curve
            plt.figure(figsize=(8, 6))
            precision, recall, _ = precision_recall_curve(self.rule_targets_test[rule].values, 
                                                         model_data['pipeline'].predict_proba(self.X_test)[:, 1])
            plt.plot(recall, precision, label=f"AUC = {metrics['pr_auc']:.3f}")
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve - Rule {rule}')
            plt.legend(loc='lower left')
            plt.savefig(os.path.join(rule_viz_dir, f'pr_curve_{rule}.png'))
            plt.close()
            
            # Confusion matrix
            conf_matrix = np.array(metrics['confusion_matrix'])
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - Rule {rule}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(os.path.join(rule_viz_dir, f'confusion_matrix_{rule}.png'))
            plt.close()
            
            # Feature importance if available
            rule_key = f"rule_{rule}"
            if rule_key in self.feature_importances:
                importance_df = self.feature_importances[rule_key]
                top_features = importance_df.head(20)
                
                plt.figure(figsize=(12, 8))
                sns.barplot(x='importance', y='feature', data=top_features)
                plt.title(f'Top 20 Feature Importances - Rule {rule}')
                plt.tight_layout()
                plt.savefig(os.path.join(rule_viz_dir, f'feature_importance_{rule}.png'))
                plt.close()
            
        except Exception as e:
            logger.error(f"Error saving visualizations for rule {rule}: {str(e)}")
    
    def _save_model_results(self):
        """Save model results to JSON."""
        try:
            # Format results for JSON serialization
            results = {}
            for name, model_data in self.models.items():
                metrics = model_data['metrics']
                results[name] = {
                    'metrics': {
                        k: v for k, v in metrics.items() 
                        if k not in ['fpr', 'tpr', 'precision', 'recall']
                    }
                }
            
            # Save results to JSON file
            with open(os.path.join(self.output_dir, 'model_results.json'), 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Model results saved to {os.path.join(self.output_dir, 'model_results.json')}")
            
        except Exception as e:
            logger.error(f"Error saving model results: {str(e)}")
    
    def save_models(self):
        """Save trained models to disk."""
        try:
            # Create models directory
            models_dir = os.path.join(self.output_dir, 'saved_models')
            os.makedirs(models_dir, exist_ok=True)
            
            # Save main models
            for name, model_data in self.models.items():
                model_path = os.path.join(models_dir, f"{name}_model.pkl")
                with open(model_path, 'wb') as f:
                    pickle.dump(model_data['pipeline'], f)
                logger.info(f"Model {name} saved to {model_path}")
            
            # Save rule-specific models
            if self.rule_models:
                rule_models_dir = os.path.join(models_dir, 'rule_models')
                os.makedirs(rule_models_dir, exist_ok=True)
                
                for rule, model_data in self.rule_models.items():
                    model_path = os.path.join(rule_models_dir, f"rule_{rule}_model.pkl")
                    with open(model_path, 'wb') as f:
                        pickle.dump(model_data['pipeline'], f)
                    logger.info(f"Rule model {rule} saved to {model_path}")
            
            logger.info("All models saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
            return False
    
    def optimize_hyperparameters(self, model_type='xgboost'):
        """
        Perform hyperparameter optimization for a specific model type.
        
        Args:
            model_type: Type of model to optimize
            
        Returns:
            dict: Best parameters and CV results
        """
        try:
            logger.info(f"Starting hyperparameter optimization for {model_type}")
            
            if model_type == 'xgboost':
                base_model = xgb.XGBClassifier(random_state=42)
                param_grid = {
                    'classifier__n_estimators': [50, 100, 200],
                    'classifier__max_depth': [3, 5, 7],
                    'classifier__learning_rate': [0.01, 0.1, 0.2],
                    'classifier__subsample': [0.8, 1.0],
                    'classifier__colsample_bytree': [0.8, 1.0],
                    'classifier__scale_pos_weight': [1, sum(self.y_train==0)/sum(self.y_train==1)]
                }
            elif model_type == 'random_forest':
                base_model = RandomForestClassifier(random_state=42)
                param_grid = {
                    'classifier__n_estimators': [50, 100, 200],
                    'classifier__max_depth': [10, 20, 30, None],
                    'classifier__min_samples_split': [2, 5, 10],
                    'classifier__min_samples_leaf': [1, 2, 4],
                    'classifier__class_weight': ['balanced', None]
                }
            elif model_type == 'lightgbm':
                base_model = lgb.LGBMClassifier(random_state=42)
                param_grid = {
                    'classifier__n_estimators': [50, 100, 200],
                    'classifier__max_depth': [3, 5, 7],
                    'classifier__learning_rate': [0.01, 0.1, 0.2],
                    'classifier__num_leaves': [31, 50, 100],
                    'classifier__class_weight': ['balanced', None]
                }
            else:
                logger.error(f"Unsupported model type for hyperparameter optimization: {model_type}")
                return None
            
            # Create pipeline with preprocessing
            preprocessor = StandardScaler()
            pipeline = ImbPipeline([
                ('preprocessor', preprocessor),
                ('classifier', base_model)
            ])
            
            # Set up grid search with cross-validation
            grid_search = GridSearchCV(
                pipeline,
                param_grid,
                cv=5,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )
            
            # Fit grid search
            grid_search.fit(self.X_train, self.y_train)
            
            # Get best parameters
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            
            logger.info(f"Best parameters for {model_type}: {best_params}")
            logger.info(f"Best cross-validation score: {best_score:.4f}")
            
            # Train model with best parameters
            best_model = grid_search.best_estimator_
            best_model.fit(self.X_train, self.y_train)
            
            # Evaluate best model
            train_score = best_model.score(self.X_train, self.y_train)
            test_score = best_model.score(self.X_test, self.y_test)
            
            # Make predictions
            y_pred = best_model.predict(self.X_test)
            y_pred_proba = best_model.predict_proba(self.X_test)[:, 1]
            
            # Calculate metrics
            classification_rep = classification_report(self.y_test, y_pred, output_dict=True)
            conf_matrix = confusion_matrix(self.y_test, y_pred)
            
            # Calculate ROC curve and AUC
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            # Calculate Precision-Recall curve
            precision, recall, _ = precision_recall_curve(self.y_test, y_pred_proba)
            pr_auc = auc(recall, precision)
            
            # Store optimized model
            self.models[f"{model_type}_optimized"] = {
                'pipeline': best_model,
                'metrics': {
                    'train_accuracy': train_score,
                    'test_accuracy': test_score,
                    'classification_report': classification_rep,
                    'confusion_matrix': conf_matrix.tolist(),
                    'roc_auc': roc_auc,
                    'pr_auc': pr_auc,
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'precision': precision.tolist(),
                    'recall': recall.tolist(),
                    'best_params': best_params,
                    'cv_score': best_score
                }
            }
            
            # Extract feature importance
            self._extract_feature_importance(f"{model_type}_optimized", best_model, self.X_train.columns)
            
            # Save optimized model results
            optimized_results = {
                'best_params': best_params,
                'cv_score': best_score,
                'test_accuracy': test_score,
                'roc_auc': roc_auc,
                'pr_auc': pr_auc,
                'classification_report': classification_rep
            }
            
            with open(os.path.join(self.output_dir, f"{model_type}_optimization_results.json"), 'w') as f:
                json.dump(optimized_results, f, indent=2)
            
            logger.info(f"Hyperparameter optimization completed for {model_type}. Results saved.")
            
            return {
                'best_params': best_params,
                'best_model': best_model,
                'cv_score': best_score,
                'test_score': test_score,
                'roc_auc': roc_auc
            }
            
        except Exception as e:
            logger.error(f"Error in hyperparameter optimization: {str(e)}")
            return None
    
    def run_complete_pipeline(self, eda=True, optimize=True, save=True):
        """
        Run the complete modeling pipeline.
        
        Args:
            eda: Whether to perform exploratory data analysis
            optimize: Whether to perform hyperparameter optimization
            save: Whether to save models
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Step 1: Load data
            if not self.load_data():
                logger.error("Failed to load data. Aborting pipeline.")
                return False
            
            # Step 2: Exploratory data analysis
            if eda:
                self.exploratory_data_analysis()
            
            # Step 3: Train models
            self.train_models()
            
            # Step 4: Train rule-specific models
            self.train_rule_specific_models()
            
            # Step 5: Hyperparameter optimization
            if optimize:
                self.optimize_hyperparameters('xgboost')  # Optimize XGBoost by default
            
            # Step 6: Save models
            if save:
                self.save_models()
            
            logger.info("Complete modeling pipeline executed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error in modeling pipeline: {str(e)}")
            return False


def main():
    """Main function to run the AML model trainer"""
    parser = argparse.ArgumentParser(description='Train machine learning models for AML detection')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing prepared ML data')
    parser.add_argument('--output_dir', type=str, help='Directory to save model outputs')
    parser.add_argument('--skip_eda', action='store_true', help='Skip exploratory data analysis')
    parser.add_argument('--skip_optimize', action='store_true', help='Skip hyperparameter optimization')
    parser.add_argument('--skip_rules', action='store_true', help='Skip rule-specific models')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = AMLModelTrainer(args.data_dir, args.output_dir)
    
    # Load data
    if not trainer.load_data():
        logger.error("Failed to load data. Exiting.")
        return
    
    # Exploratory data analysis
    if not args.skip_eda:
        trainer.exploratory_data_analysis()
    
    # Train models
    trainer.train_models()
    
    # Train rule-specific models
    if not args.skip_rules:
        trainer.train_rule_specific_models()
    
    # Hyperparameter optimization
    if not args.skip_optimize:
        trainer.optimize_hyperparameters('xgboost')
    
    # Save models
    trainer.save_models()
    
    logger.info("AML model training completed successfully")


if __name__ == "__main__":
    main()