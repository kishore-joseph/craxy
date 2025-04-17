import os
import sys
import pandas as pd
import numpy as np
import json
import pickle
import logging
import argparse
import mysql.connector
from mysql.connector import Error
import configparser
from datetime import datetime, timedelta
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("aml_model_inference.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("AML_MODEL_INFERENCE")

class AMLModelInference:
    """
    Inference class for using trained AML detection models to identify suspicious transactions.
    """
    
    def __init__(self, model_dir, config_file=None):
        """
        Initialize the inference engine.
        
        Args:
            model_dir: Directory containing the trained models
            config_file: Path to database configuration file (optional)
        """
        self.model_dir = model_dir
        self.config_file = config_file
        
        self.main_model = None
        self.rule_models = {}
        self.feature_columns = []
        
        # Load configuration if provided
        if config_file:
            self.db_config = self._load_config()
            self.connection = None
        
    def _load_config(self):
        """
        Load database configuration from the config file.
        
        Returns:
            dict: Database configuration parameters
        """
        logger.info(f"Loading configuration from {self.config_file}")
        
        config = configparser.ConfigParser()
        config.read(self.config_file)
        
        db_config = {
            'host': config.get('DEFAULT', 'db.host'),
            'user': config.get('DEFAULT', 'db.username'),
            'password': config.get('DEFAULT', 'db.password'),
            'database': config.get('DEFAULT', 'db.name'),
            'port': config.getint('DEFAULT', 'db.port'),
            'connection_timeout': config.getint('DEFAULT', 'db.connection_timeout')
        }
        
        return db_config
    
    def connect_to_database(self):
        """
        Establish connection to the MySQL database.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        if not self.config_file:
            logger.error("No configuration file provided for database connection")
            return False
            
        try:
            logger.info(f"Connecting to database {self.db_config['database']} at {self.db_config['host']}")
            
            self.connection = mysql.connector.connect(
                host=self.db_config['host'],
                user=self.db_config['user'],
                password=self.db_config['password'],
                database=self.db_config['database'],
                port=self.db_config['port'],
                connection_timeout=self.db_config['connection_timeout']
            )
            
            if self.connection.is_connected():
                logger.info("Successfully connected to MySQL database")
                return True
            else:
                logger.error("Failed to connect to MySQL database")
                return False
                
        except Error as e:
            logger.error(f"Error connecting to MySQL database: {str(e)}")
            return False
    
    def load_models(self):
        """
        Load the trained models from disk.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load main model
            main_model_path = os.path.join(self.model_dir, 'saved_models', 'xgboost_optimized_model.pkl')
            if not os.path.exists(main_model_path):
                main_model_path = os.path.join(self.model_dir, 'saved_models', 'xgboost_model.pkl')
            
            if not os.path.exists(main_model_path):
                # Try other models
                model_files = [f for f in os.listdir(os.path.join(self.model_dir, 'saved_models')) 
                              if f.endswith('_model.pkl') and not f.startswith('rule_')]
                if model_files:
                    main_model_path = os.path.join(self.model_dir, 'saved_models', model_files[0])
                else:
                    logger.error("No main model found")
                    return False
            
            with open(main_model_path, 'rb') as f:
                self.main_model = pickle.load(f)
            
            logger.info(f"Main model loaded from {main_model_path}")
            
            # Load feature columns
            features_path = os.path.join(self.model_dir, 'feature_columns.json')
            if os.path.exists(features_path):
                with open(features_path, 'r') as f:
                    self.feature_columns = json.load(f)
                logger.info(f"Feature columns loaded: {len(self.feature_columns)} features")
            
            # Load rule-specific models
            rule_models_dir = os.path.join(self.model_dir, 'saved_models', 'rule_models')
            if os.path.exists(rule_models_dir):
                rule_model_files = [f for f in os.listdir(rule_models_dir) if f.endswith('_model.pkl')]
                
                for model_file in rule_model_files:
                    rule_name = model_file.replace('rule_', '').replace('_model.pkl', '')
                    
                    with open(os.path.join(rule_models_dir, model_file), 'rb') as f:
                        self.rule_models[rule_name] = pickle.load(f)
                
                logger.info(f"Loaded {len(self.rule_models)} rule-specific models")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return False
    
    def prepare_features(self, transaction_data, account_data=None, entity_data=None):
        """
        Prepare features for model inference.
        
        Args:
            transaction_data: DataFrame of new transactions
            account_data: DataFrame of account information (optional)
            entity_data: DataFrame of entity information (optional)
            
        Returns:
            DataFrame: Prepared features for model inference
        """
        try:
            # Make a copy of transactions
            features = transaction_data.copy()
            
            # Add account features if available
            if account_data is not None:
                # Merge on account_id
                features = features.merge(account_data, on='account_id', how='left', suffixes=('', '_account'))
            
            # Add entity features if available
            if entity_data is not None:
                # Merge on entity_id
                features = features.merge(entity_data, on='entity_id', how='left', suffixes=('', '_entity'))
            
            # Convert timestamp to datetime and extract features
            if 'timestamp' in features.columns:
                features['timestamp'] = pd.to_datetime(features['timestamp'])
                features['hour'] = features['timestamp'].dt.hour
                features['day_of_week'] = features['timestamp'].dt.dayofweek
                features['month'] = features['timestamp'].dt.month
                features['day'] = features['timestamp'].dt.day
                features['year'] = features['timestamp'].dt.year
                
                # Flag transactions outside normal banking hours (8am-6pm)
                features['outside_banking_hours'] = ((features['hour'] < 8) | 
                                                   (features['hour'] > 18)).astype(int)
                
                # Flag weekend transactions
                features['is_weekend'] = (features['day_of_week'] > 4).astype(int)
            
            # Define a function to check for round amounts
            if 'amount' in features.columns:
                features['is_round_amount'] = (features['amount'] % 100 == 0).astype(int)
                features['just_below_threshold'] = ((features['amount'] >= 9000) & 
                                                   (features['amount'] < 10000)).astype(int)
            
            # Handle missing values
            # For numeric columns, fill with 0
            numeric_cols = features.select_dtypes(include=['number']).columns
            features[numeric_cols] = features[numeric_cols].fillna(0)
            
            # For categorical columns, fill with 'Unknown'
            categorical_cols = features.select_dtypes(include=['object']).columns
            features[categorical_cols] = features[categorical_cols].fillna('Unknown')
            
            # If we have feature columns from training, make sure we match those
            if self.feature_columns:
                # One-hot encode categorical columns
                features_encoded = pd.get_dummies(features)
                
                # Ensure all expected columns are present
                for col in self.feature_columns:
                    if col not in features_encoded.columns:
                        features_encoded[col] = 0
                
                # Select only the expected columns in the right order
                features_final = features_encoded[self.feature_columns]
            else:
                # Remove non-feature columns
                drop_cols = [
                    'transaction_id', 'account_id', 'entity_id', 'timestamp', 
                    'created_at', 'updated_at', 'description', 'counterparty_id', 
                    'counterparty_name', 'counterparty_bank'
                ]
                
                # Drop columns that should not be used as features
                features_final = features.drop(columns=[col for col in drop_cols if col in features.columns])
                
                # One-hot encode remaining categorical columns
                features_final = pd.get_dummies(features_final)
                
                # Save the feature columns for future use
                self.feature_columns = features_final.columns.tolist()
                with open(os.path.join(self.model_dir, 'feature_columns.json'), 'w') as f:
                    json.dump(self.feature_columns, f)
            
            return features_final
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            return None
    
    def predict(self, features, threshold=0.5):
        """
        Make predictions using the main model.
        
        Args:
            features: DataFrame of prepared features
            threshold: Probability threshold for classification
            
        Returns:
            tuple: (predictions, probabilities)
        """
        try:
            if self.main_model is None:
                if not self.load_models():
                    logger.error("Failed to load main model for prediction")
                    return None, None
            
            # Make predictions
            probabilities = self.main_model.predict_proba(features)[:, 1]
            predictions = (probabilities >= threshold).astype(int)
            
            return predictions, probabilities
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            return None, None
    
    def predict_rules(self, features):
        """
        Make predictions using rule-specific models.
        
        Args:
            features: DataFrame of prepared features
            
        Returns:
            dict: Dictionary of rule predictions and probabilities
        """
        try:
            if not self.rule_models:
                logger.warning("No rule-specific models loaded")
                return {}
            
            rule_predictions = {}
            
            for rule, model in self.rule_models.items():
                # Make predictions for this rule
                rule_proba = model.predict_proba(features)[:, 1]
                rule_pred = (rule_proba >= 0.5).astype(int)
                
                rule_predictions[rule] = {
                    'predictions': rule_pred,
                    'probabilities': rule_proba
                }
            
            return rule_predictions
            
        except Exception as e:
            logger.error(f"Error making rule predictions: {str(e)}")
            return {}
    
    def analyze_transactions(self, transactions, accounts=None, entities=None):
        """
        Analyze transactions and identify suspicious activity.
        
        Args:
            transactions: DataFrame of transactions to analyze
            accounts: DataFrame of account information (optional)
            entities: DataFrame of entity information (optional)
            
        Returns:
            DataFrame: Transactions with added prediction and risk score
        """
        try:
            # Prepare features
            features = self.prepare_features(transactions, accounts, entities)
            
            if features is None:
                logger.error("Failed to prepare features for inference")
                return transactions
            
            # Make predictions
            predictions, probabilities = self.predict(features)
            
            if predictions is None:
                logger.error("Failed to make predictions")
                return transactions
            
            # Make rule-specific predictions
            rule_predictions = self.predict_rules(features)
            
            # Add predictions to transactions
            results = transactions.copy()
            results['is_suspicious'] = predictions
            results['risk_score'] = probabilities
            
            # Add rule-specific predictions
            for rule, rule_data in rule_predictions.items():
                results[f'{rule}_flag'] = rule_data['predictions']
                results[f'{rule}_score'] = rule_data['probabilities']
            
            # Generate reasons for suspicious transactions
            results['alert_reasons'] = ""
            suspicious_idx = results[results['is_suspicious'] == 1].index
            
            for idx in suspicious_idx:
                reasons = []
                
                # Check main model score
                score = results.loc[idx, 'risk_score']
                if score >= 0.9:
                    reasons.append(f"Very high risk score ({score:.2f})")
                elif score >= 0.7:
                    reasons.append(f"High risk score ({score:.2f})")
                
                # Check rule-specific flags
                for rule, rule_data in rule_predictions.items():
                    if rule_data['predictions'][idx] == 1:
                        rule_score = rule_data['probabilities'][idx]
                        rule_name = rule.replace('TM_', 'TM-')
                        reasons.append(f"{rule_name} ({rule_score:.2f})")
                
                # Set alert reasons
                results.loc[idx, 'alert_reasons'] = "; ".join(reasons)
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing transactions: {str(e)}")
            return transactions
    
    def get_recent_transactions(self, days=1):
        """
        Get recent transactions from the database.
        
        Args:
            days: Number of days back to fetch transactions
            
        Returns:
            tuple: (transactions, accounts, entities)
        """
        if not self.connection or not self.connection.is_connected():
            if not self.connect_to_database():
                logger.error("Failed to connect to database")
                return None, None, None
        
        try:
            cursor = self.connection.cursor(dictionary=True)
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Query transactions
            transaction_query = """
            SELECT * FROM transactions 
            WHERE timestamp >= %s AND timestamp <= %s
            """
            cursor.execute(transaction_query, (start_date, end_date))
            transactions = cursor.fetchall()
            
            if not transactions:
                logger.info(f"No transactions found in the last {days} days")
                return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            
            # Convert to DataFrame
            df_transactions = pd.DataFrame(transactions)
            
            # Get unique account IDs and entity IDs
            account_ids = df_transactions['account_id'].unique().tolist()
            entity_ids = df_transactions['entity_id'].unique().tolist() if 'entity_id' in df_transactions.columns else []
            
            # Query account information
            df_accounts = pd.DataFrame()
            if account_ids:
                placeholders = ', '.join(['%s'] * len(account_ids))
                account_query = f"""
                SELECT * FROM account
                WHERE account_id IN ({placeholders})
                """
                cursor.execute(account_query, account_ids)
                accounts = cursor.fetchall()
                if accounts:
                    df_accounts = pd.DataFrame(accounts)
            
            # Query entity information
            df_entities = pd.DataFrame()
            if entity_ids:
                # Try to get customer entities first
                placeholders = ', '.join(['%s'] * len(entity_ids))
                customer_query = f"""
                SELECT * FROM entity_customer
                WHERE entity_id IN ({placeholders})
                """
                cursor.execute(customer_query, entity_ids)
                customers = cursor.fetchall()
                
                # Then try to get business entities
                business_query = f"""
                SELECT * FROM entity_business
                WHERE entity_id IN ({placeholders})
                """
                cursor.execute(business_query, entity_ids)
                businesses = cursor.fetchall()
                
                # Combine customer and business entities
                entities = []
                if customers:
                    for customer in customers:
                        customer['entity_type'] = 'customer'
                        entities.append(customer)
                
                if businesses:
                    for business in businesses:
                        business['entity_type'] = 'business'
                        entities.append(business)
                
                if entities:
                    df_entities = pd.DataFrame(entities)
            
            cursor.close()
            logger.info(f"Retrieved {len(df_transactions)} transactions from database")
            
            return df_transactions, df_accounts, df_entities
            
        except Error as e:
            logger.error(f"Error retrieving transactions from database: {str(e)}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    def save_alerts(self, suspicious_transactions):
        """
        Save suspicious transactions to alerts table in the database.
        
        Args:
            suspicious_transactions: DataFrame of suspicious transactions
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.connection or not self.connection.is_connected():
            if not self.connect_to_database():
                logger.error("Failed to connect to database")
                return False
        
        try:
            cursor = self.connection.cursor()
            
            # Check if alerts table exists, create if not
            create_alerts_table = """
            CREATE TABLE IF NOT EXISTS aml_alerts (
                alert_id INT AUTO_INCREMENT PRIMARY KEY,
                transaction_id VARCHAR(255) NOT NULL,
                risk_score FLOAT NOT NULL,
                alert_reasons TEXT,
                status VARCHAR(50) DEFAULT 'Open',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                resolved_at TIMESTAMP NULL,
                INDEX idx_transaction_id (transaction_id),
                INDEX idx_status (status)
            )
            """
            cursor.execute(create_alerts_table)
            self.connection.commit()
            
            # Filter only suspicious transactions
            alerts = suspicious_transactions[suspicious_transactions['is_suspicious'] == 1]
            
            if alerts.empty:
                logger.info("No suspicious transactions to save as alerts")
                return True
            
            # Insert alerts
            alert_count = 0
            for _, alert in alerts.iterrows():
                # Check if alert already exists
                check_query = "SELECT alert_id FROM aml_alerts WHERE transaction_id = %s"
                cursor.execute(check_query, (alert['transaction_id'],))
                existing = cursor.fetchone()
                
                if not existing:
                    # Insert new alert
                    insert_query = """
                    INSERT INTO aml_alerts (transaction_id, risk_score, alert_reasons)
                    VALUES (%s, %s, %s)
                    """
                    cursor.execute(insert_query, (
                        alert['transaction_id'],
                        float(alert['risk_score']),
                        alert.get('alert_reasons', '')
                    ))
                    alert_count += 1
            
            self.connection.commit()
            cursor.close()
            
            logger.info(f"Saved {alert_count} new alerts to database")
            return True
            
        except Error as e:
            logger.error(f"Error saving alerts to database: {str(e)}")
            return False
    
    def run_scheduled_analysis(self, interval_minutes=60, days_back=1):
        """
        Run scheduled analysis of recent transactions.
        
        Args:
            interval_minutes: Time between analyses in minutes
            days_back: Number of days back to analyze
            
        Returns:
            None
        """
        logger.info(f"Starting scheduled analysis every {interval_minutes} minutes, looking back {days_back} days")
        
        try:
            while True:
                # Load models if not loaded
                if self.main_model is None:
                    if not self.load_models():
                        logger.error("Failed to load models. Retrying in 5 minutes.")
                        time.sleep(300)
                        continue
                
                # Get recent transactions
                transactions, accounts, entities = self.get_recent_transactions(days=days_back)
                
                if not transactions.empty:
                    # Analyze transactions
                    results = self.analyze_transactions(transactions, accounts, entities)
                    
                    # Save alerts
                    suspicious_count = results['is_suspicious'].sum()
                    if suspicious_count > 0:
                        logger.info(f"Found {suspicious_count} suspicious transactions")
                        self.save_alerts(results)
                    else:
                        logger.info("No suspicious transactions found")
                
                # Wait for next interval
                logger.info(f"Analysis complete. Next analysis in {interval_minutes} minutes.")
                time.sleep(interval_minutes * 60)
                
        except KeyboardInterrupt:
            logger.info("Scheduled analysis stopped by user")
        except Exception as e:
            logger.error(f"Error in scheduled analysis: {str(e)}")
    
    def batch_analyze_from_file(self, file_path, output_path=None):
        """
        Analyze transactions from a CSV file.
        
        Args:
            file_path: Path to CSV file containing transactions
            output_path: Path to save results (optional)
            
        Returns:
            DataFrame: Analysis results
        """
        try:
            # Load transactions
            transactions = pd.read_csv(file_path)
            logger.info(f"Loaded {len(transactions)} transactions from {file_path}")
            
            # Load accounts and entities if available
            accounts = None
            entities = None
            
            accounts_path = os.path.join(os.path.dirname(file_path), 'accounts.csv')
            if os.path.exists(accounts_path):
                accounts = pd.read_csv(accounts_path)
                logger.info(f"Loaded {len(accounts)} accounts from {accounts_path}")
            
            entities_path = os.path.join(os.path.dirname(file_path), 'entities.csv')
            if os.path.exists(entities_path):
                entities = pd.read_csv(entities_path)
                logger.info(f"Loaded {len(entities)} entities from {entities_path}")
            
            # Analyze transactions
            results = self.analyze_transactions(transactions, accounts, entities)
            
            # Save results if output path provided
            if output_path:
                results.to_csv(output_path, index=False)
                logger.info(f"Results saved to {output_path}")
                
                # Also save a summary of suspicious transactions
                suspicious = results[results['is_suspicious'] == 1]
                if not suspicious.empty:
                    summary_path = output_path.replace('.csv', '_suspicious.csv')
                    suspicious.to_csv(summary_path, index=False)
                    logger.info(f"Suspicious transactions saved to {summary_path}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing transactions from file: {str(e)}")
            return pd.DataFrame()


def main():
    """Main function to run the AML model inference"""
    parser = argparse.ArgumentParser(description='Run inference using trained AML models')
    parser.add_argument('--model_dir', type=str, required=True, help='Directory containing trained models')
    parser.add_argument('--config', type=str, help='Path to database configuration file')
    parser.add_argument('--mode', type=str, choices=['batch', 'schedule', 'once'], default='once',
                       help='Mode to run inference (batch: from file, schedule: periodic, once: one-time)')
    parser.add_argument('--file', type=str, help='Path to CSV file for batch mode')
    parser.add_argument('--output', type=str, help='Path to save results for batch mode')
    parser.add_argument('--interval', type=int, default=60, help='Interval in minutes for scheduled mode')
    parser.add_argument('--days', type=int, default=1, help='Days back to analyze for scheduled/once mode')
    
    args = parser.parse_args()
    
    # Create inference engine
    engine = AMLModelInference(args.model_dir, args.config)
    
    # Load models
    if not engine.load_models():
        logger.error("Failed to load models. Exiting.")
        return
    
    # Run in specified mode
    if args.mode == 'batch':
        if not args.file:
            logger.error("File path must be provided for batch mode")
            return
        
        engine.batch_analyze_from_file(args.file, args.output)
        
    elif args.mode == 'schedule':
        engine.run_scheduled_analysis(args.interval, args.days)
        
    else:  # once
        transactions, accounts, entities = engine.get_recent_transactions(days=args.days)
        
        if transactions.empty:
            logger.info(f"No transactions found in the last {args.days} days")
            return
        
        results = engine.analyze_transactions(transactions, accounts, entities)
        
        suspicious_count = results['is_suspicious'].sum()
        if suspicious_count > 0:
            logger.info(f"Found {suspicious_count} suspicious transactions")
            engine.save_alerts(results)
        else:
            logger.info("No suspicious transactions found")
            
        # Save results if output path provided
        if args.output:
            results.to_csv(args.output, index=False)
            logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()