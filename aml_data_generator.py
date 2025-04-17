import os
import sys
import pandas as pd
import numpy as np
import mysql.connector
from mysql.connector import Error
import configparser
from datetime import datetime, timedelta, date
import random
from tqdm import tqdm
import uuid
import json
import time
import logging
from faker import Faker
import re
import ipaddress
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("aml_data_generator.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("AML_DATA_GENERATOR")

# Initialize Faker for generating realistic data
fake = Faker()

class AMLTrainingDataGenerator:
    """
    Generates synthetic training data for AML machine learning models based on 
    existing database schema.
    """
    
    def __init__(self, config_file):
        """
        Initialize the data generator with database configuration.
        
        Args:
            config_file (str): Path to the configuration file
        """
        self.config_file = config_file
        self.db_config = self._load_config()
        self.connection = None
        self.schema_info = {}
        self.generated_data = {}
        self.entity_map = {}  # Maps entity_id to entity type and details
        self.account_map = {}  # Maps account_id to entity_id
        
        # AML specific configurations
        self.high_risk_countries = [
            'AF', 'KP', 'IR', 'RU', 'BY', 'MM', 'CU', 'SY', 'VE', 'IQ',
            'LY', 'SO', 'SS', 'ZW', 'YE'
        ]
        
        self.high_risk_industries = [
            'Gambling', 'Casino', 'Cryptocurrency', 'Virtual Currency', 
            'Money Services Business', 'Cash Intensive Business',
            'Offshore Banking', 'Shell Company', 'Art Dealer', 'Precious Metals',
            'Arms Dealer', 'Adult Entertainment'
        ]
        
    def read_properties_file(self,file_path):
        properties = {}
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    try:
                        key_value = line.split('=', 1)
                        if len(key_value) == 2:
                            properties[key_value[0].strip()] = key_value[1].strip()
                    except Exception as e:
                        print(f"Warning: Could not parse line: {line}")
        return properties
    
    def _load_config(self):
        """
        Load database configuration from the config file.
        
        Returns:
            dict: Database configuration parameters
        """
        logger.info(f"Loading configuration from {self.config_file}")
        
        config = configparser.ConfigParser()
        # config.read(self.config_file)
        config = self.read_properties_file('aml_config.properties')
        
        db_config = {
            'host': config.get('db.host'),
            'user': config.get('db.username'),
            'password': config.get('db.password'),
            'database': config.get('db.name'),
            'port': '3306'

        }
        
        return db_config
    
    
    
    def connect_to_database(self):
        """
        Establish connection to the MySQL database.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            logger.info(f"Connecting to database {self.db_config['database']} at {self.db_config['host']}")
            
            self.connection = mysql.connector.connect(
                host=self.db_config['host'],
                user=self.db_config['user'],
                password=self.db_config['password'],
                database=self.db_config['database'],
                port=int(self.db_config['port'])
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
        
    def _safe_date_max(self, date1, date2):
        """
        Safely compare and return the later of two dates, even if they have different types
        (date vs datetime).
        
        Args:
            date1: A date or datetime object
            date2: A date or datetime object
            
        Returns:
            The later of the two dates, as a datetime object
        """
        # Import datetime at method level to avoid conflicts
        from datetime import datetime
        
        # Handle date1 conversion if needed
        if hasattr(date1, 'date'):  # It's already a datetime
            dt1 = date1
        else:  # It's a date, convert to datetime
            dt1 = datetime(date1.year, date1.month, date1.day)
            
        # Handle date2 conversion if needed
        if hasattr(date2, 'date'):  # It's already a datetime
            dt2 = date2
        else:  # It's a date, convert to datetime
            dt2 = datetime(date2.year, date2.month, date2.day)
            
        # Return the later datetime
        return dt1 if dt1 > dt2 else dt2
    
    def get_table_schema(self, table_name):
        """
        Retrieve the schema information for a specific table.
        
        Args:
            table_name (str): Name of the table
            
        Returns:
            list: List of dictionaries containing column information
        """
        try:
            if not self.connection or not self.connection.is_connected():
                self.connect_to_database()
                
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute(f"DESCRIBE {table_name}")
            schema = cursor.fetchall()
            cursor.close()
            
            return schema
            
        except Error as e:
            logger.error(f"Error retrieving schema for table {table_name}: {str(e)}")
            return []
    
    def load_schema_information(self):
        """
        Load schema information for all required tables.
        
        Returns:
            bool: True if successful, False otherwise
        """
        required_tables = [
            'entity_customer', 
            'entity_business', 
            'entity_dependents',
            'beneficiary', 
            'account', 
            'transactions'
        ]
        
        try:
            for table in required_tables:
                schema = self.get_table_schema(table)
                if schema:
                    self.schema_info[table] = schema
                    logger.info(f"Successfully loaded schema for {table}")
                else:
                    logger.warning(f"Failed to load schema for {table}")
            
            if len(self.schema_info) == len(required_tables):
                logger.info("Successfully loaded schema information for all required tables")
                return True
            else:
                logger.warning("Failed to load schema information for all required tables")
                return False
                
        except Exception as e:
            logger.error(f"Error loading schema information: {str(e)}")
            return False
    
    def generate_consumer_data(self, num_records=1000):
        """
        Generate synthetic consumer data.
        
        Args:
            num_records (int): Number of consumer records to generate
            
        Returns:
            DataFrame: Generated consumer data
        """
        logger.info(f"Generating {num_records} consumer records")

        # Connect to the database
        if not hasattr(self, 'connection') or not self.connection.is_connected():
            if not self.connect_to_database():
                logger.error("Failed to connect to database")
                return pd.DataFrame()
        
        cursor = self.connection.cursor(dictionary=True)

        # Get potential records from database
        cursor.execute("""SELECT ENTITY_ID, FIRST_NAME, LAST_NAME, DATE_OF_BIRTH, GENDER, NATIONALITY, CUSTOMERTYPE, PEP, INSERTED_DATE,UPDATED_DATE FROM entity_customer LIMIT %s""", (num_records,))
            
        customer_records = cursor.fetchall()
        
        consumers = []

        for i, customer in enumerate(customer_records):
            entity_id = customer['ENTITY_ID']
            first_name = customer['FIRST_NAME']
            last_name = customer['LAST_NAME']
            dob = customer['DATE_OF_BIRTH']
            gender=customer['GENDER']
            # Foreign customer attributes
            is_foreign = 'Y' if customer['NATIONALITY'] != 'US' else 'N'
            country_code = customer['NATIONALITY']
            customer_type = customer['CUSTOMERTYPE']
            # PEP (Politically Exposed Person) status
            is_pep = customer['PEP']
            insert_date = customer['INSERTED_DATE']
            update_date = customer['UPDATED_DATE'] 

            is_high_risk = random.random() < 0.05  # 5% of accounts are high_risk           
            
            
            # Create the consumer record
            consumer = {
                'entity_id': entity_id,
                'first_name': first_name,
                'last_name': last_name,
                'full_name': f"{first_name} {last_name}",
                'dob': dob,
                'gender': gender,
                'country_code': country_code,
                'is_foreign': is_foreign,
                'is_high_risk': is_high_risk,
                'is_pep': is_pep,
                'created_at': insert_date,
                'updated_at': update_date
            }
            
            consumers.append(consumer)
            
            # Store in entity map
            self.entity_map[entity_id] = {
                'type': 'consumer',
                'details': consumer
            }
        
        df_consumers = pd.DataFrame(consumers)
        self.generated_data['entity_customer'] = df_consumers
        
        logger.info(f"Successfully generated {len(df_consumers)} consumer records")
        return df_consumers
    
    def generate_business_data(self, num_records=1000):
        """
        Generate synthetic business data.
        
        Args:
            num_records (int): Number of business records to generate
            
        Returns:
            DataFrame: Generated business data
        """
        logger.info(f"Generating {num_records} business records")

        # Connect to the database
        if not hasattr(self, 'connection') or not self.connection.is_connected():
            if not self.connect_to_database():
                logger.error("Failed to connect to database")
                return pd.DataFrame()
        
        cursor = self.connection.cursor(dictionary=True)

        try:

            # Fetch existing business types from the database
            cursor.execute("SELECT DISTINCT business_type FROM entity_business")
            db_business_types = [row['business_type'] for row in cursor.fetchall()]
            business_types = db_business_types if db_business_types else ['LLC', 'Corporation', 'Partnership', 'Sole Proprietorship', 'Non-profit']

            high_risk_industrys = set(["445131",'445310','457110','722511','722513','722514','713210','713290','441110','441120','441330','812930','458310','423940','522390','459991','458210','444110','444120','444230','459110','459120','459130','444240','459140','445132','459310','459410','449110','459420','449121','459510','449129','459999','449210','561510','811111','811114','811198','441210','441222','441227'])

            businesses = []
            # Get potential records from database
            cursor.execute("""
                        select eb.ENTITY_ID, eb.LEGAL_NAME, eb.business_type, eb.date_of_incorporation,eb.TAX_EXEMPT, DOING_BUSINESS_AS, ei.IDENTIFIER_VALUE as naics_code
                        from entity_business eb
                        left join entity_identifier ei on eb.ENTITY_ID  = ei.ENTITY_ID 
                        where ei.identifier_type = 'NAICS'
                        LIMIT %s
            """, (num_records,))
                
            business_records = cursor.fetchall()
            
            for i, business in enumerate(business_records):
                entity_id = business['ENTITY_ID']
                legal_name = business['LEGAL_NAME']
                doing_business_as = business['DOING_BUSINESS_AS']
                business_type = business['business_type']
                business_activity = ''
                date_of_incorporation =business['date_of_incorporation']
                naics_code = business['naics_code']
                # Risk-based attributes (preserving your logic)
                is_high_risk_industry = naics_code in high_risk_industrys
                
                # Create the business record
                business = {
                    'ENTITY_BUSINESS_ID': None,  # Auto-increment field
                    'ENTITY_ID': entity_id,
                    'LEGAL_NAME': legal_name,
                    'BUSINESS_TYPE': business_type,
                    'DATE_OF_INCORPORATION': date_of_incorporation,
                    'BUSINESS_ACTIVITY': business_activity,
                    'DOING_BUSINESS_AS': doing_business_as,
                    'NAICS_CODE': naics_code,
                    'IS_HIGH_RISK_INDUSTRY':is_high_risk_industry
                }
                
                businesses.append(business)
                
                # Store in entity map
                self.entity_map[entity_id] = {
                    'type': 'business',
                    'details': business
                }

        except Error as e:
            logger.error(f"Database error while generating businesses: {str(e)}")
        finally:
            cursor.close()
        
        df_businesses = pd.DataFrame(businesses)
        self.generated_data['entity_business'] = df_businesses
        
        logger.info(f"Successfully generated {len(df_businesses)} business records")
        return df_businesses
    
    def generate_dependent_data(self, num_per_customer=0.05):
        """
        Generate synthetic dependent data for consumers.
        
        Args:
            num_per_customer (float): Average number of dependents per customer
            
        Returns:
            DataFrame: Generated dependent data
        """
        if 'entity_customer' not in self.generated_data:
            logger.error("Cannot generate dependents: consumer data not generated yet")
            return pd.DataFrame()
        
        consumers = self.generated_data['entity_customer']
        logger.info(f"Generating dependents for {len(consumers)} consumers")
        
        # Connect to the database
        if not hasattr(self, 'connection') or not self.connection.is_connected():
            if not self.connect_to_database():
                logger.error("Failed to connect to database")
                return pd.DataFrame()
        
        dependents = []
        cursor = self.connection.cursor(dictionary=True)
        
        try:
            for _, consumer in tqdm(consumers.iterrows(), total=len(consumers), desc="Generating dependents"):
                # Skip some consumers
                if random.random() > num_per_customer * 3:  # Adjust probability to get average num_per_customer
                    continue
                
                # Generate 1-3 dependents for this consumer
                num_deps = random.randint(1, 3)
                
                # Get relationship types from database
                cursor.execute("SELECT DISTINCT dependent_relationship_type FROM entity_dependents")
                relationship_types = [row['dependent_relationship_type'] for row in cursor.fetchall()]
                
                # Use default relationships if none found in database
                if not relationship_types:
                    relationship_types = ['Child', 'Sibling', 'Guardian', 'Parent', 'Spouse']
                
                for _ in range(num_deps):
                    # Calculate appropriate age range based on relationship
                    consumer_age = datetime.now().year - consumer['dob'].year
                    
                    # Get a person from the database with appropriate age
                    if relation := random.choice(relationship_types) == 'Child':
                        # Child should be younger
                        cursor.execute("""
                            SELECT * FROM entity_dependents 
                            WHERE YEAR(CURDATE()) - YEAR(dependent_date_of_birth) BETWEEN 1 AND 25
                            ORDER BY RAND() 
                            LIMIT 1
                        """)
                    elif relation == 'Spouse':
                        # Spouse should be of similar age
                        min_age = max(18, consumer_age - 15)
                        max_age = min(90, consumer_age + 15)
                        cursor.execute("""
                            SELECT * FROM entity_dependents 
                            WHERE YEAR(CURDATE()) - YEAR(dependent_date_of_birth) BETWEEN %s AND %s
                            ORDER BY RAND() 
                            LIMIT 1
                        """, (min_age, max_age))
                    elif relation == 'Parent':
                        # Parent should be older
                        min_age = consumer_age + 15
                        max_age = consumer_age + 50
                        cursor.execute("""
                            SELECT * FROM entity_dependents 
                            WHERE YEAR(CURDATE()) - YEAR(dependent_date_of_birth) BETWEEN %s AND %s
                            ORDER BY RAND() 
                            LIMIT 1
                        """, (min_age, max_age))
                    elif relation == 'Guardian':
                        # Guardian should be older - similar to parent but with wider age range
                        min_age = consumer_age + 10  # Could be slightly younger than parent
                        max_age = consumer_age + 60  # Could be older than parent
                        cursor.execute("""
                            SELECT * FROM entity_dependents 
                            WHERE YEAR(CURDATE()) - YEAR(dependent_date_of_birth) BETWEEN %s AND %s
                            ORDER BY RAND() 
                            LIMIT 1
                        """, (min_age, max_age))
                    else:  # Sibling
                        # Sibling should be of similar age
                        min_age = max(1, consumer_age - 15)
                        max_age = min(90, consumer_age + 15)
                        cursor.execute("""
                            SELECT * FROM entity_dependents 
                            WHERE YEAR(CURDATE()) - YEAR(dependent_date_of_birth) BETWEEN %s AND %s
                            ORDER BY RAND() 
                            LIMIT 1
                        """, (min_age, max_age))
                    
                    person = cursor.fetchone()
                    
                    # If no suitable person found, use default values
                    if person:
                        dependent_name = f"{person['DEPENDENT_NAME']}"
                        dob = person['dependent_date_of_birth']
                        # Use consumer's last name for family members
                        if relation in ['Child', 'Spouse', 'Sibling', 'Guardian']:
                            dependent_name = f"{person['dependent_name']}"
                        # Generate email and phone for dependent
                        dependent_email = person['dependent_email']
                        dependent_phone = person['dependent_phone']
                    
                    dependent = {
                        'ENTITY_DEPENDENT_ID': None,  # Auto-increment field
                        'ENTITY_ID': consumer['entity_id'],
                        'DEPENDENT_NAME': dependent_name,
                        'dependent_date_of_birth': dob,
                        'dependent_relationship_type': relation,
                        'dependent_phone': dependent_phone,
                        'dependent_email': dependent_email
                    }
                    
                    dependents.append(dependent)
                    
        except Error as e:
            logger.error(f"Database error while generating dependents: {str(e)}")
        finally:
            cursor.close()
        
        df_dependents = pd.DataFrame(dependents)
        self.generated_data['entity_dependents'] = df_dependents
        
        logger.info(f"Successfully generated {len(df_dependents)} dependents")
        return df_dependents
    
    def generate_beneficiary_data(self, num_per_business=0.5):
        """
        Generate synthetic beneficiary data for businesses.
        
        Args:
            num_per_business (float): Average number of beneficiaries per business
            
        Returns:
            DataFrame: Generated beneficiary data
        """
        if 'entity_business' not in self.generated_data:
            logger.error("Cannot generate beneficiaries: business data not generated yet")
            return pd.DataFrame()
        
        businesses = self.generated_data['entity_business']
        logger.info(f"Generating beneficiaries for {len(businesses)} businesses")
        
        beneficiaries = []
        cursor = self.connection.cursor(dictionary=True)
        
        try:
            for _, business in tqdm(businesses.iterrows(), total=len(businesses), desc="Generating beneficiaries"):
                # Skip some businesses
                if random.random() > num_per_business * 2:  # Adjust probability to get average num_per_business
                    continue
                
                # Generate 1-5 beneficiaries for this business
                num_benes = random.randint(1, 5)
                
                # Get potential beneficiaries from database
                cursor.execute("""
                    select b.BENEFICIARY_ID , b.BENEFICIARY_FIRST_NAME ,b.BENEFICIARY_LAST_NAME , 
                               eo.ownership_percentage ,eo.ownership_type, eo.insert_date as created_at, eo.update_date as updated_at
                    from entity_ownership eo
                    join beneficiary b on eo.OWNED_ENTITY_ID = b.BENEFICIARY_ID 
                    order by b.PARTY_ACCOUNT_RELATIVE_ORDER
                """, (num_benes,))
                
                person_records = cursor.fetchall()
                
                for i, person in enumerate(person_records):
                    beneficiary_id = person['beneficiary_id']
                    first_name = person['beneficiary_first_name']
                    last_name = person['beneficiary_last_name']
                    ownership_pct = person['ownership_percentage']
                    role = person['ownership_type']
                    
                    
                    # Risk factors - check if person is in pep table
                    is_pep = 'N'
                    is_high_risk_country = 'N'
                    
                    is_high_risk = is_pep or is_high_risk_country
                    
                    # Foreign status
                    is_foreign = 'N' if person['beneficiary_nationality'] == 'US' else 'Y'
                    country_code = person['beneficiary_nationality']
                    
                    beneficiary = {
                        'beneficiary_id': beneficiary_id,
                        'entity_id': business['entity_id'],
                        'first_name': first_name,
                        'last_name': last_name,
                        'full_name': f"{first_name} {last_name}",
                        'dob': person['date_of_birth'],
                        'role': role,
                        'ownership_percentage': ownership_pct,
                        'is_pep': is_pep,
                        'is_high_risk': is_high_risk,
                        'country_code': country_code,
                        'is_foreign': is_foreign,
                        'created_at': person['created_at'],
                        'updated_at': person['updated_at']
                    }
                    
                    beneficiaries.append(beneficiary)
        
        except Error as e:
            logger.error(f"Database error while generating beneficiaries: {str(e)}")
        finally:
            cursor.close()
        
        df_beneficiaries = pd.DataFrame(beneficiaries)
        self.generated_data['beneficiary'] = df_beneficiaries
        
        logger.info(f"Successfully generated {len(df_beneficiaries)} beneficiaries")
        return df_beneficiaries
    
    def generate_account_data(self):
        """
        Generate synthetic account data for entities.
        
        Args:
            avg_accounts_per_entity (float): Average number of accounts per entity
            
        Returns:
            DataFrame: Generated account data
        """

        logger.info("Retrieving account data from database")
    
        # Establish database connection using existing method
        if not hasattr(self, 'connection') or not self.connection.is_connected():
            connection_success = self.connect_to_database()
            if not connection_success:
                logger.error("Cannot retrieve accounts: database connection failed")
                return pd.DataFrame()
        
        try:
            # Create cursor
            cursor = self.connection.cursor(dictionary=True)
            
            # Query to fetch account data with entity information
            query = """
                SELECT 
                    a.account_id, 
                    a.entity_id, 
                    a.account_type, 
                    a.account_status, 
                    a.open_date, 
                    a.account_balance, 
                    a.account_currency, 
                    '' as is_high_risk, 
                    a.last_activity_date, 
                    CASE WHEN a.account_status != 'Active' 
                        THEN DATEDIFF(CURRENT_DATE, a.last_activity_date) 
                        ELSE 0 
                    END as days_inactive,
                    a.insert_date as created_at,
                    a.update_date as updated_at
                FROM account a
                JOIN entity e ON a.entity_id = e.entity_id
            """
            
            # Execute query
            cursor.execute(query)
            
            # Fetch all results
            account_records = cursor.fetchall()
            
            # Convert to DataFrame
            accounts = pd.DataFrame(account_records)
            
            # Build the account map for later use
            self.account_map = {}
            
            # Get entity types in a separate query for efficiency
            cursor.execute("SELECT entity_id, entity_type FROM entity")
            entity_types = {row['entity_id']: row['entity_type'] for row in cursor.fetchall()}
            
            for _, row in accounts.iterrows():
                account_id = row['account_id']
                entity_id = row['entity_id']
                account_type = row['account_type']
                entity_type = entity_types.get(entity_id, 'unknown')
                status = row['account_status']
                # Account opening information
                account_opening_date = row['open_date']
                
                # Last activity date
                days_since_opening = (datetime.now().date() - account_opening_date).days
                is_dormant = random.random() < 0.05  # 5% of accounts are dormant
                
                if is_dormant:
                    last_activity_date = account_opening_date + timedelta(days=random.randint(1, min(90, days_since_opening)))
                else:
                    last_activity_date = fake.date_between(start_date=account_opening_date, end_date='today')
                
                
                self.account_map[account_id] = {
                    'entity_id': entity_id,
                    'account_id':account_id,
                    'entity_type': entity_type,
                    'account_type': account_type,
                    'opening_date': account_opening_date,
                    'status': status,
                    'is_dormant' : is_dormant,
                    'last_activity_date' : last_activity_date
                }
            
            # Store in the generated_data dictionary for consistency with other methods
            self.generated_data['account'] = accounts
            
            cursor.close()
            
            logger.info(f"Successfully retrieved {len(accounts)} accounts")
            return accounts
            
        except Error as e:
            logger.error(f"Error retrieving account data: {str(e)}")
            return pd.DataFrame()
    
    def generate_transaction_data(self, num_transactions=50000, suspicious_pct=0.05):
        """
        Generate synthetic transaction data for accounts.
        
        Args:
            num_transactions (int): Number of transactions to generate
            suspicious_pct (float): Percentage of transactions that should be suspicious
            
        Returns:
            tuple: (DataFrame of transactions, DataFrame of labels)
        """

        if not self.account_map:
            logger.error("Cannot retrieve transactions: account map is empty")
            return pd.DataFrame(), pd.DataFrame()
        
        logger.info("Retrieving transactions from database")

        # Establish database connection using existing method
        if not hasattr(self, 'connection') or not self.connection.is_connected():
            connection_success = self.connect_to_database()
            if not connection_success:
                logger.error("Cannot retrieve transactions: database connection failed")
                return pd.DataFrame(), pd.DataFrame()
            
        try:
            # Create cursor
            cursor = self.connection.cursor(dictionary=True)
            
            # Query to fetch transaction data
            transaction_query = """
                SELECT 
                    t.transaction_id,
                    t.account_id,
                    a.entity_id,
                    t.transaction_type,
                    t.amount,
                    t.currency,
                    t.transaction_date as timestamp,
                    t.description,
                    '' as status,
                    ec.ENTITY_ID as counterparty_id, concat(ec.FIRST_NAME, ec.LAST_NAME) as counterparty_name , rn.institution_name as counterparty_bank ,
                    '' as country,
                    '' as ip_address,
                    t.insert_date as created_at,
                    t.update_date as updated_at
                FROM transactions t
                inner JOIN account a ON t.account_id = a.account_id
                left join account a1 on t.related_account = a1.account_number
                join entity_customer ec on a1.ENTITY_ID = ec.ENTITY_ID 
                join routing_numbers rn on a1.account_routing_number = rn.routing_number 
				union 
				SELECT 
                    t.transaction_id,
                    t.account_id,
                    a.entity_id,
                    t.transaction_type,
                    t.amount,
                    t.currency,
                    t.transaction_date as timestamp,
                    t.description,
                    '' as status,
                    ec.ENTITY_ID as counterparty_id, concat(ec.legal_name) as counterparty_name , rn.institution_name as counterparty_bank ,
                    '' as country,
                    '' as ip_address,
                    t.insert_date as created_at,
                    t.update_date as updated_at
                FROM transactions t
                inner JOIN account a ON t.account_id = a.account_id
                left join account a2 on t.related_account = a2.account_number
                join entity_business ec on a2.ENTITY_ID = ec.ENTITY_ID 
                join routing_numbers rn on a2.account_routing_number = rn.routing_number 
                ORDER BY timestamp
            """
        
            # Execute query
            cursor.execute(transaction_query)
            
            # Fetch all transaction records
            transaction_records = cursor.fetchall()
            
            # Convert to DataFrame
            df_transactions = pd.DataFrame(transaction_records)
            
            # Since there's no aml_labels table, create an empty DataFrame with the appropriate columns
            aml_label_columns = [
                'transaction_id', 'is_suspicious', 
                'TM-001', 'TM-001a', 'TM-002', 'TM-003', 'TM-004',
                'TM-006', 'TM-007', 'TM-008', 'TM-009', 'TM-010',
                'TM-011', 'TM-012', 'TM-013', 'TM-014', 'TM-015',
                'TM-018', 'TM-019', 'TM-020', 'TM-021', 'TM-022',
                'TM-025', 'TM-026', 'TM-029', 'TM-030', 'TM-031',
                'TM-032', 'TM-033', 'TM-035', 'TM-036', 'TM-037',
                'TM-039', 'TM-040', 'TM-041'
            ]
            
            # Create empty labels DataFrame
            df_labels = pd.DataFrame(columns=aml_label_columns)
            
            # If needed, you can populate with default values (all transactions marked as non-suspicious)
            if not df_transactions.empty:
                labels_data = []
                for txn_id in df_transactions['transaction_id']:
                    label_record = {'transaction_id': txn_id, 'is_suspicious': 0}
                    # Set all rule flags to 0
                    for col in aml_label_columns:
                        if col not in ['transaction_id', 'is_suspicious']:
                            label_record[col] = 0
                    labels_data.append(label_record)
                
                df_labels = pd.DataFrame(labels_data)
            
            # Update account tracking for potential later use
            account_txn_count = defaultdict(int)
            account_last_txn_date = {}
            
            for _, row in df_transactions.iterrows():
                account_id = row['account_id']
                account_txn_count[account_id] += 1
                account_last_txn_date[account_id] = row['timestamp']
            
            # Store the data
            self.generated_data['transactions'] = df_transactions
            self.generated_data['aml_labels'] = df_labels
            
            cursor.close()
            
            logger.info(f"Successfully retrieved {len(df_transactions)} transactions")
            return df_transactions, df_labels
            
        except Error as e:
            logger.error(f"Error retrieving transaction data: {str(e)}")
            return pd.DataFrame(), pd.DataFrame()
        
    def _generate_transaction_description(self, transaction_type):
        """Generate a realistic transaction description based on type"""
        if transaction_type == 'Deposit':
            return random.choice([
                "Deposit", "Branch Deposit", "Mobile Deposit", "Direct Deposit", "Payroll Deposit"
            ])
        elif transaction_type == 'Withdrawal':
            return random.choice([
                "Withdrawal", "Branch Withdrawal", "ATM Withdrawal", "Cash Withdrawal"
            ])
        elif transaction_type == 'Transfer':
            return random.choice([
                "Transfer", "Account Transfer", "Internal Transfer", "Fund Transfer"
            ])
        elif transaction_type == 'Payment':
            return random.choice([
                f"Payment to {fake.company()}", f"Bill Payment - {fake.company()}", 
                "Credit Card Payment", "Loan Payment", "Utility Payment"
            ])
        elif transaction_type == 'Purchase':
            return random.choice([
                f"Purchase - {fake.company()}", "Retail Purchase", "Online Purchase", 
                f"POS Purchase - {fake.company()}"
            ])
        elif transaction_type == 'Cash Deposit':
            return "Cash Deposit"
        elif transaction_type == 'Cash Withdrawal':
            return "Cash Withdrawal"
        elif transaction_type == 'Wire Transfer In':
            return f"Incoming Wire - {fake.company()}"
        elif transaction_type == 'Wire Transfer Out':
            return f"Outgoing Wire - {fake.company()}"
        elif transaction_type == 'ACH Credit':
            return f"ACH Credit - {fake.company()}"
        elif transaction_type == 'ACH Debit':
            return f"ACH Debit - {fake.company()}"
        elif transaction_type == 'Check Deposit':
            return "Check Deposit"
        elif transaction_type == 'Check Payment':
            return f"Check Payment - {fake.random_number(digits=4)}"
        elif transaction_type == 'ATM Withdrawal':
            return f"ATM Withdrawal - {fake.city()}"
        elif transaction_type == 'POS Purchase':
            return f"POS Purchase - {fake.company()}"
        elif transaction_type == 'Online Payment':
            return f"Online Payment - {fake.company()}"
        elif transaction_type == 'Mobile Deposit':
            return "Mobile Deposit"
        elif transaction_type == 'International Transfer In':
            return f"International Wire In - {fake.country()}"
        elif transaction_type == 'International Transfer Out':
            return f"International Wire Out - {fake.country()}"
        else:
            return "Transaction"
    
    def _generate_large_cash_transaction(self):
        """Generate a large cash transaction (TM-001, TM-001a)"""
        # Randomly select an account (preferably an MSB account for TM-001a)
        msb_accounts = [
            aid for aid, ainfo in self.account_map.items() 
            if self.entity_map[ainfo['entity_id']]['type'] == 'business' and
            self.entity_map[ainfo['entity_id']]['details'].get('is_msb', False)
        ]
        
        if msb_accounts and random.random() < 0.6:
            # 60% chance to use an MSB account if available
            account_id = random.choice(msb_accounts)
            account_info = self.account_map[account_id]
            is_msb = True
        else:
            # Otherwise use any account
            account_id = random.choice(list(self.account_map.keys()))
            account_info = self.account_map[account_id]
            is_msb = False
        
        entity_id = account_info['entity_id']
        entity_type = account_info['entity_type']
        account_type = account_info['account_type']
        account_open_date = account_info['opening_date']
        
        # Get entity details
        entity_details = self.entity_map[entity_id]['details']
        
        # Generate transaction details
        transaction_type = random.choice(['Cash Deposit', 'Cash Withdrawal'])
        
        # Large amount
        if is_msb:
            # For MSB accounts, anything over $2,500 is suspicious (TM-001a)
            amount = random.uniform(2500, 15000)
        else:
            # For regular accounts, anything over $10,000 is suspicious (TM-001)
            amount = random.uniform(10000, 50000)
        
        # Round to dollars (many large cash transactions are in whole dollars)
        amount = round(amount, 0)
        
        one_year_ago = datetime.now() - timedelta(days=365)
    
        # Handle the type comparison correctly
        if hasattr(one_year_ago, 'date'):  # It's a datetime object
            # Convert account_open_date to datetime if it's a date
            if not hasattr(account_open_date, 'date'):  # It's a date object
                effective_start_date = datetime(
                    account_open_date.year,
                    account_open_date.month,
                    account_open_date.day,
                    0, 0, 0
                ) 
            else:
                effective_start_date = account_open_date
                
            # Compare and use the later date
            if effective_start_date > one_year_ago:
                start_date_for_fake = effective_start_date
            else:
                start_date_for_fake = one_year_ago
        else:
            # Fallback if datetime.now() somehow returns a date
            start_date_for_fake = one_year_ago
        
        timestamp = fake.date_time_between(
            start_date=start_date_for_fake,
            end_date=datetime.now()
        )
        
        # # Generate timestamp (business hours)
        # timestamp = fake.date_time_between(
        #     start_date=max(account_open_date, datetime.now() - timedelta(days=365)),
        #     end_date=datetime.now()
        # )
        hour = random.randint(9, 16)  # 9 AM to 4 PM
        timestamp = timestamp.replace(hour=hour, minute=random.randint(0, 59))
        
        # Create transaction record
        transaction = {
            'transaction_id': str(uuid.uuid4()),
            'account_id': account_id,
            'entity_id': entity_id,
            'transaction_type': transaction_type,
            'amount': amount,
            'currency': 'USD',
            'timestamp': timestamp,
            'description': transaction_type,
            'status': 'Completed',
            'counterparty_id': None,
            'counterparty_name': None,
            'counterparty_bank': None,
            'country': entity_details.get('country_code', 'US'),
            'ip_address': str(ipaddress.IPv4Address(random.randint(0, 2**32-1))),
            'created_at': timestamp,
            'updated_at': timestamp
        }
        
        # Create label record
        label = {
            'transaction_id': transaction['transaction_id'],
            'is_suspicious': 1,
            'TM-001': 1 if amount >= 10000 else 0,
            'TM-001a': 1 if is_msb and amount >= 2500 else 0,
            'TM-022': 1 if entity_details.get('is_cash_intensive', False) else 0
        }
        
        # Set all other flags to 0
        for rule in [
            'TM-002', 'TM-003', 'TM-004', 'TM-006', 'TM-007', 'TM-008', 'TM-009', 'TM-010',
            'TM-011', 'TM-012', 'TM-013', 'TM-014', 'TM-015', 'TM-018', 'TM-019', 'TM-020',
            'TM-021', 'TM-025', 'TM-026', 'TM-029', 'TM-030', 'TM-031', 'TM-032', 'TM-033',
            'TM-035', 'TM-036', 'TM-037', 'TM-039', 'TM-040', 'TM-041'
        ]:
            label[rule] = 0
        
        return transaction, label
    
    def _generate_cash_structuring_transaction(self):
        """Generate a structuring transaction (TM-002, TM-026)"""
        account_id = random.choice(list(self.account_map.keys()))
        account_info = self.account_map[account_id]
        entity_id = account_info['entity_id']
        entity_type = account_info['entity_type']
        account_open_date = account_info['opening_date']
        
        # Get entity details
        entity_details = self.entity_map[entity_id]['details']
        
        # Generate transaction details
        transaction_type = random.choice(['Cash Deposit', 'Cash Withdrawal'])
        
        # Amount just below reporting threshold (typical structuring behavior)
        amount = random.uniform(8700, 9900)
        
        # Round to dollars or cents
        if random.random() < 0.7:
            # 70% chance of whole dollars
            amount = round(amount, 0)
        else:
            amount = round(amount, 2)
        
        # Generate timestamp (business hours)
        timestamp = fake.date_time_between(
            start_date=self._safe_date_max(account_open_date, datetime.now() - timedelta(days=365)),
            end_date=datetime.now()
        )
        hour = random.randint(9, 16)  # 9 AM to 4 PM
        timestamp = timestamp.replace(hour=hour, minute=random.randint(0, 59))
        
        # Create transaction record
        transaction = {
            'transaction_id': str(uuid.uuid4()),
            'account_id': account_id,
            'entity_id': entity_id,
            'transaction_type': transaction_type,
            'amount': amount,
            'currency': 'USD',
            'timestamp': timestamp,
            'description': transaction_type,
            'status': 'Completed',
            'counterparty_id': None,
            'counterparty_name': None,
            'counterparty_bank': None,
            'country': entity_details.get('country_code', 'US'),
            'ip_address': str(ipaddress.IPv4Address(random.randint(0, 2**32-1))),
            'created_at': timestamp,
            'updated_at': timestamp
        }
        
        # Create label record
        label = {
            'transaction_id': transaction['transaction_id'],
            'is_suspicious': 1,
            'TM-002': 1,  # Structuring
            'TM-026': 1,  # Just Below Threshold
        }
        
        # Set all other flags to 0
        for rule in [
            'TM-001', 'TM-001a', 'TM-003', 'TM-004', 'TM-006', 'TM-007', 'TM-008', 'TM-009', 'TM-010',
            'TM-011', 'TM-012', 'TM-013', 'TM-014', 'TM-015', 'TM-018', 'TM-019', 'TM-020',
            'TM-021', 'TM-022', 'TM-025', 'TM-029', 'TM-030', 'TM-031', 'TM-032', 'TM-033',
            'TM-035', 'TM-036', 'TM-037', 'TM-039', 'TM-040', 'TM-041'
        ]:
            label[rule] = 0
        
        return transaction, label
    
    def _generate_rapid_movement_transaction(self):
        """Generate a rapid movement transaction (TM-003, TM-018, TM-019)"""
        account_id = random.choice(list(self.account_map.keys()))
        account_info = self.account_map[account_id]
        entity_id = account_info['entity_id']
        entity_type = account_info['entity_type']
        account_open_date = account_info['opening_date']
        
        # Get entity details
        entity_details = self.entity_map[entity_id]['details']
        
        # Determine which specific pattern to generate
        pattern_type = random.choice(['rapid_movement', 'rapid_wires', 'round_trip'])
        
        if pattern_type == 'rapid_movement':
            # Rapid movement of funds (transfer in followed by transfer out)
            transaction_type = random.choice(['Wire Transfer In', 'ACH Credit'])
            amount = random.uniform(5000, 50000)
            # Set TM-003 flag
            rule_flags = {'TM-003': 1, 'TM-018': 0, 'TM-019': 0}
            
        elif pattern_type == 'rapid_wires':
            # Rapid succession of wires
            transaction_type = random.choice(['Wire Transfer Out', 'Wire Transfer In'])
            amount = random.uniform(5000, 50000)
            # Set TM-018 flag
            rule_flags = {'TM-003': 0, 'TM-018': 1, 'TM-019': 0}
            
        else:  # round_trip
            # Same-day round trip (transfer out and in for similar amounts)
            transaction_type = random.choice(['Wire Transfer Out', 'ACH Debit'])
            amount = random.uniform(5000, 50000)
            # Set TM-019 flag
            rule_flags = {'TM-003': 0, 'TM-018': 0, 'TM-019': 1}
        
        # Generate timestamp
        timestamp = fake.date_time_between(
            start_date=self._safe_date_max(account_open_date, datetime.now() - timedelta(days=365)),
            end_date=datetime.now()
        )
        
        # Create transaction record
        transaction = {
            'transaction_id': str(uuid.uuid4()),
            'account_id': account_id,
            'entity_id': entity_id,
            'transaction_type': transaction_type,
            'amount': round(amount, 2),
            'currency': 'USD',
            'timestamp': timestamp,
            'description': self._generate_transaction_description(transaction_type),
            'status': 'Completed',
            'counterparty_id': None,
            'counterparty_name': fake.company(),
            'counterparty_bank': fake.company() + " Bank",
            'country': entity_details.get('country_code', 'US'),
            'ip_address': str(ipaddress.IPv4Address(random.randint(0, 2**32-1))),
            'created_at': timestamp,
            'updated_at': timestamp
        }
        
        # Create label record
        label = {
            'transaction_id': transaction['transaction_id'],
            'is_suspicious': 1,
            'TM-003': rule_flags['TM-003'],
            'TM-018': rule_flags['TM-018'],
            'TM-019': rule_flags['TM-019']
        }
        
        # Set all other flags to 0
        for rule in [
            'TM-001', 'TM-001a', 'TM-002', 'TM-004', 'TM-006', 'TM-007', 'TM-008', 'TM-009', 'TM-010',
            'TM-011', 'TM-012', 'TM-013', 'TM-014', 'TM-015', 'TM-020',
            'TM-021', 'TM-022', 'TM-025', 'TM-026', 'TM-029', 'TM-030', 'TM-031', 'TM-032', 'TM-033',
            'TM-035', 'TM-036', 'TM-037', 'TM-039', 'TM-040', 'TM-041'
        ]:
            label[rule] = 0
        
        return transaction, label
    
    def _generate_multiple_currency_transaction(self):
        """Generate a multiple currency transaction (TM-004)"""
        account_id = random.choice(list(self.account_map.keys()))
        account_info = self.account_map[account_id]
        entity_id = account_info['entity_id']
        entity_type = account_info['entity_type']
        account_open_date = account_info['opening_date']
        
        # Get entity details
        entity_details = self.entity_map[entity_id]['details']
        
        # Generate transaction details
        transaction_type = random.choice([
            'Wire Transfer In', 'Wire Transfer Out', 
            'International Transfer In', 'International Transfer Out'
        ])
        
        # Generate non-USD currency
        currency = random.choice(['EUR', 'GBP', 'CAD', 'JPY', 'AUD', 'CHF', 'SGD'])
        
        # Generate amount
        amount = random.uniform(1000, 50000)
        
        # Generate timestamp
        timestamp = fake.date_time_between(
            start_date=self._safe_date_max(account_open_date, datetime.now() - timedelta(days=365)),
            end_date=datetime.now()
        )
        
        # Create transaction record
        transaction = {
            'transaction_id': str(uuid.uuid4()),
            'account_id': account_id,
            'entity_id': entity_id,
            'transaction_type': transaction_type,
            'amount': round(amount, 2),
            'currency': currency,
            'timestamp': timestamp,
            'description': self._generate_transaction_description(transaction_type),
            'status': 'Completed',
            'counterparty_id': None,
            'counterparty_name': fake.company(),
            'counterparty_bank': fake.company() + " Bank",
            'country': random.choice(['GB', 'DE', 'FR', 'IT', 'ES', 'JP', 'CH', 'SG']),
            'ip_address': str(ipaddress.IPv4Address(random.randint(0, 2**32-1))),
            'created_at': timestamp,
            'updated_at': timestamp
        }
        
        # Create label record
        label = {
            'transaction_id': transaction['transaction_id'],
            'is_suspicious': 1,
            'TM-004': 1,  # Multiple Currency
            'TM-031': 1 if 'International' in transaction_type else 0  # Cross-Border Wire
        }
        
        # Set all other flags to 0
        for rule in [
            'TM-001', 'TM-001a', 'TM-002', 'TM-003', 'TM-006', 'TM-007', 'TM-008', 'TM-009', 'TM-010',
            'TM-011', 'TM-012', 'TM-013', 'TM-014', 'TM-015', 'TM-018', 'TM-019', 'TM-020',
            'TM-021', 'TM-022', 'TM-025', 'TM-026', 'TM-029', 'TM-030', 'TM-032', 'TM-033',
            'TM-035', 'TM-036', 'TM-037', 'TM-039', 'TM-040', 'TM-041'
        ]:
            if rule != 'TM-031':  # Skip if already set
                label[rule] = 0
        
        return transaction, label
    
    def _generate_round_dollar_transaction(self):
        """Generate a round dollar transaction (TM-007)"""
        account_id = random.choice(list(self.account_map.keys()))
        account_info = self.account_map[account_id]
        entity_id = account_info['entity_id']
        entity_type = account_info['entity_type']
        account_open_date = account_info['opening_date']
        
        # Get entity details
        entity_details = self.entity_map[entity_id]['details']
        
        # Generate transaction details
        transaction_type = random.choice([
            'Wire Transfer Out', 'Wire Transfer In', 'Cash Deposit', 'Cash Withdrawal'
        ])
        
        # Generate round dollar amount (multiple of 1000)
        base = random.randint(5, 50)
        amount = base * 1000.00
        
        # Generate timestamp
        timestamp = fake.date_time_between(
            start_date=self._safe_date_max(account_open_date, datetime.now() - timedelta(days=365)),
            end_date=datetime.now()
        )
        
        # Create transaction record
        transaction = {
            'transaction_id': str(uuid.uuid4()),
            'account_id': account_id,
            'entity_id': entity_id,
            'transaction_type': transaction_type,
            'amount': amount,
            'currency': 'USD',
            'timestamp': timestamp,
            'description': self._generate_transaction_description(transaction_type),
            'status': 'Completed',
            'counterparty_id': None,
            'counterparty_name': fake.company() if 'Wire' in transaction_type else None,
            'counterparty_bank': fake.company() + " Bank" if 'Wire' in transaction_type else None,
            'country': entity_details.get('country_code', 'US'),
            'ip_address': str(ipaddress.IPv4Address(random.randint(0, 2**32-1))),
            'created_at': timestamp,
            'updated_at': timestamp
        }
        
        # Create label record
        label = {
            'transaction_id': transaction['transaction_id'],
            'is_suspicious': 1,
            'TM-007': 1,  # Round Dollar
        }
        
        # Set TM-001 if it's a large cash transaction
        if transaction_type in ['Cash Deposit', 'Cash Withdrawal'] and amount >= 10000:
            label['TM-001'] = 1
        else:
            label['TM-001'] = 0
            
        # Set TM-001a if it's an MSB account with cash transaction > 2500
        is_msb = (entity_type == 'business' and entity_details.get('is_msb', False))
        if transaction_type in ['Cash Deposit', 'Cash Withdrawal'] and is_msb and amount >= 2500:
            label['TM-001a'] = 1
        else:
            label['TM-001a'] = 0
        
        # Set all other flags to 0
        for rule in [
            'TM-002', 'TM-003', 'TM-004', 'TM-006', 'TM-008', 'TM-009', 'TM-010',
            'TM-011', 'TM-012', 'TM-013', 'TM-014', 'TM-015', 'TM-018', 'TM-019', 'TM-020',
            'TM-021', 'TM-022', 'TM-025', 'TM-026', 'TM-029', 'TM-030', 'TM-031', 'TM-032', 'TM-033',
            'TM-035', 'TM-036', 'TM-037', 'TM-039', 'TM-040', 'TM-041'
        ]:
            label[rule] = 0
        
        return transaction, label
    
    def _generate_unusual_hours_transaction(self):
        """Generate a transaction during unusual hours (TM-008)"""
        account_id = random.choice(list(self.account_map.keys()))
        account_info = self.account_map[account_id]
        entity_id = account_info['entity_id']
        entity_type = account_info['entity_type']
        account_open_date = account_info['opening_date']
        
        # Get entity details
        entity_details = self.entity_map[entity_id]['details']
        
        # Generate transaction details
        transaction_type = random.choice([
            'Online Payment', 'POS Purchase', 'ATM Withdrawal', 'Transfer', 
            'Wire Transfer Out', 'Wire Transfer In'
        ])
        
        # Generate amount
        amount = random.uniform(200, 5000)
        
        # Generate unusual hours timestamp
        timestamp = fake.date_time_between(
            start_date=self._safe_date_max(account_open_date, datetime.now() - timedelta(days=365)),
            end_date=datetime.now()
        )
        
        # Set unusual hour (11 PM to 5 AM)
        unusual_hours = [23, 0, 1, 2, 3, 4, 5]
        hour = random.choice(unusual_hours)
        timestamp = timestamp.replace(hour=hour, minute=random.randint(0, 59))
        
        # Create transaction record
        transaction = {
            'transaction_id': str(uuid.uuid4()),
            'account_id': account_id,
            'entity_id': entity_id,
            'transaction_type': transaction_type,
            'amount': round(amount, 2),
            'currency': 'USD',
            'timestamp': timestamp,
            'description': self._generate_transaction_description(transaction_type),
            'status': 'Completed',
            'counterparty_id': None,
            'counterparty_name': fake.company() if transaction_type in ['Wire Transfer In', 'Wire Transfer Out'] else None,
            'counterparty_bank': fake.company() + " Bank" if transaction_type in ['Wire Transfer In', 'Wire Transfer Out'] else None,
            'country': entity_details.get('country_code', 'US'),
            'ip_address': str(ipaddress.IPv4Address(random.randint(0, 2**32-1))),
            'created_at': timestamp,
            'updated_at': timestamp
        }
        
        # Create label record
        label = {
            'transaction_id': transaction['transaction_id'],
            'is_suspicious': 1,
            'TM-008': 1,  # Unusual Hours
        }
        
        # Set all other flags to 0
        for rule in [
            'TM-001', 'TM-001a', 'TM-002', 'TM-003', 'TM-004', 'TM-006', 'TM-007', 'TM-009', 'TM-010',
            'TM-011', 'TM-012', 'TM-013', 'TM-014', 'TM-015', 'TM-018', 'TM-019', 'TM-020',
            'TM-021', 'TM-022', 'TM-025', 'TM-026', 'TM-029', 'TM-030', 'TM-031', 'TM-032', 'TM-033',
            'TM-035', 'TM-036', 'TM-037', 'TM-039', 'TM-040', 'TM-041'
        ]:
            label[rule] = 0
        
        return transaction, label
    
    def _generate_high_risk_transaction(self):
        """Generate a transaction from a high-risk entity (TM-010, TM-011, TM-020)"""
        # Find high-risk entities
        high_risk_entity_ids = [
            entity_id for entity_id, entity_info in self.entity_map.items()
            if entity_info['details'].get('is_high_risk', False)
        ]
        
        # If no high-risk entities, create transaction from any entity
        if not high_risk_entity_ids:
            account_id = random.choice(list(self.account_map.keys()))
        else:
            # Find accounts belonging to high-risk entities
            high_risk_accounts = [
                account_id for account_id, account_info in self.account_map.items()
                if account_info['entity_id'] in high_risk_entity_ids
            ]
            
            if high_risk_accounts:
                account_id = random.choice(high_risk_accounts)
            else:
                account_id = random.choice(list(self.account_map.keys()))
        
        # Get account and entity info
        account_info = self.account_map[account_id]
        entity_id = account_info['entity_id']
        entity_type = account_info['entity_type']
        account_open_date = account_info['opening_date']
        
        # Get entity details
        entity_details = self.entity_map[entity_id]['details']
        
        # Choose which high-risk rule to trigger
        is_shell_company = entity_type == 'business' and entity_details.get('is_shell_company', False)
        is_high_risk_industry = entity_type == 'business' and entity_details.get('is_high_risk_industry', False)
        is_high_risk_customer = entity_details.get('is_high_risk', False)
        
        if is_shell_company:
            rule_flags = {'TM-010': 0, 'TM-011': 1, 'TM-020': 0}
        elif is_high_risk_industry:
            rule_flags = {'TM-010': 0, 'TM-011': 0, 'TM-020': 1}
        else:
            rule_flags = {'TM-010': 1, 'TM-011': 0, 'TM-020': 0}
        
        # Generate transaction details
        if entity_type == 'business':
            transaction_type = random.choice([
                'Wire Transfer In', 'Wire Transfer Out', 'ACH Credit', 'ACH Debit', 
                'Cash Deposit', 'Cash Withdrawal'
            ])
        else:
            transaction_type = random.choice([
                'Wire Transfer In', 'Wire Transfer Out', 'Transfer', 
                'Cash Deposit', 'Cash Withdrawal'
            ])
        
        # Generate amount
        amount = random.uniform(1000, 20000)
        
        # Generate timestamp
        timestamp = fake.date_time_between(
            start_date=self._safe_date_max(account_open_date, datetime.now() - timedelta(days=365)),
            end_date=datetime.now()
        )
        
        # Create transaction record
        transaction = {
            'transaction_id': str(uuid.uuid4()),
            'account_id': account_id,
            'entity_id': entity_id,
            'transaction_type': transaction_type,
            'amount': round(amount, 2),
            'currency': 'USD',
            'timestamp': timestamp,
            'description': self._generate_transaction_description(transaction_type),
            'status': 'Completed',
            'counterparty_id': None,
            'counterparty_name': fake.company() if transaction_type in ['Wire Transfer In', 'Wire Transfer Out'] else None,
            'counterparty_bank': fake.company() + " Bank" if transaction_type in ['Wire Transfer In', 'Wire Transfer Out'] else None,
            'country': entity_details.get('country_code', 'US'),
            'ip_address': str(ipaddress.IPv4Address(random.randint(0, 2**32-1))),
            'created_at': timestamp,
            'updated_at': timestamp
        }
        
        # Create label record
        label = {
            'transaction_id': transaction['transaction_id'],
            'is_suspicious': 1,
            'TM-010': rule_flags['TM-010'],
            'TM-011': rule_flags['TM-011'],
            'TM-020': rule_flags['TM-020'],
        }
        
        # Set TM-001 if it's a large cash transaction
        if transaction_type in ['Cash Deposit', 'Cash Withdrawal'] and amount >= 10000:
            label['TM-001'] = 1
        else:
            label['TM-001'] = 0
            
        # Set TM-001a if it's an MSB account with cash transaction > 2500
        is_msb = (entity_type == 'business' and entity_details.get('is_msb', False))
        if transaction_type in ['Cash Deposit', 'Cash Withdrawal'] and is_msb and amount >= 2500:
            label['TM-001a'] = 1
        else:
            label['TM-001a'] = 0
            
        # Set TM-022 if it's a cash-intensive business
        if entity_type == 'business' and entity_details.get('is_cash_intensive', False):
            label['TM-022'] = 1
        else:
            label['TM-022'] = 0
        
        # Set all other flags to 0
        for rule in [
            'TM-002', 'TM-003', 'TM-004', 'TM-006', 'TM-007', 'TM-008', 'TM-009',
            'TM-012', 'TM-013', 'TM-014', 'TM-015', 'TM-018', 'TM-019',
            'TM-021', 'TM-025', 'TM-026', 'TM-029', 'TM-030', 'TM-031', 'TM-032', 'TM-033',
            'TM-035', 'TM-036', 'TM-037', 'TM-039', 'TM-040', 'TM-041'
        ]:
            label[rule] = 0
        
        return transaction, label
    
    def _generate_dormant_account_transaction(self):
        """Generate a transaction on a dormant account (TM-013)"""
        # Find dormant accounts
        dormant_accounts = [
            account_id for account_id, account_info in self.account_map.items()
            if account_info['status'] == 'Dormant'
        ]
        
        # If no dormant accounts, create a transaction from any account
        if not dormant_accounts:
            account_id = random.choice(list(self.account_map.keys()))
        else:
            account_id = random.choice(dormant_accounts)
        
        # Get account and entity info
        account_info = self.account_map[account_id]
        entity_id = account_info['entity_id']
        entity_type = account_info['entity_type']
        account_open_date = account_info['opening_date']
        
        # Get entity details
        entity_details = self.entity_map[entity_id]['details']
        
        # Generate transaction details
        transaction_type = random.choice([
            'Cash Deposit', 'Wire Transfer In', 'Wire Transfer Out',
            'Transfer', 'ACH Credit', 'ACH Debit'
        ])
        
        # Generate amount
        amount = random.uniform(1000, 15000)
        
        # Generate timestamp
        timestamp = fake.date_time_between(
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now()
        )
        
        # Create transaction record
        transaction = {
            'transaction_id': str(uuid.uuid4()),
            'account_id': account_id,
            'entity_id': entity_id,
            'transaction_type': transaction_type,
            'amount': round(amount, 2),
            'currency': 'USD',
            'timestamp': timestamp,
            'description': self._generate_transaction_description(transaction_type),
            'status': 'Completed',
            'counterparty_id': None,
            'counterparty_name': fake.company() if transaction_type in ['Wire Transfer In', 'Wire Transfer Out'] else None,
            'counterparty_bank': fake.company() + " Bank" if transaction_type in ['Wire Transfer In', 'Wire Transfer Out'] else None,
            'country': entity_details.get('country_code', 'US'),
            'ip_address': str(ipaddress.IPv4Address(random.randint(0, 2**32-1))),
            'created_at': timestamp,
            'updated_at': timestamp
        }
        
        # Create label record
        label = {
            'transaction_id': transaction['transaction_id'],
            'is_suspicious': 1,
            'TM-013': 1,  # Dormant Account
        }
        
        # Set TM-001 if it's a large cash transaction
        if transaction_type in ['Cash Deposit', 'Cash Withdrawal'] and amount >= 10000:
            label['TM-001'] = 1
        else:
            label['TM-001'] = 0
            
        # Set TM-001a if it's an MSB account with cash transaction > 2500
        is_msb = (entity_type == 'business' and entity_details.get('is_msb', False))
        if transaction_type in ['Cash Deposit', 'Cash Withdrawal'] and is_msb and amount >= 2500:
            label['TM-001a'] = 1
        else:
            label['TM-001a'] = 0
        
        # Set all other flags to 0
        for rule in [
            'TM-002', 'TM-003', 'TM-004', 'TM-006', 'TM-007', 'TM-008', 'TM-009', 'TM-010',
            'TM-011', 'TM-012', 'TM-014', 'TM-015', 'TM-018', 'TM-019', 'TM-020',
            'TM-021', 'TM-022', 'TM-025', 'TM-026', 'TM-029', 'TM-030', 'TM-031', 'TM-032', 'TM-033',
            'TM-035', 'TM-036', 'TM-037', 'TM-039', 'TM-040', 'TM-041'
        ]:
            label[rule] = 0
        
        return transaction, label
    
    def _generate_unusual_frequency_transaction(self):
        """Generate a transaction with unusual frequency (TM-009)"""
        account_id = random.choice(list(self.account_map.keys()))
        account_info = self.account_map[account_id]
        entity_id = account_info['entity_id']
        entity_type = account_info['entity_type']
        account_open_date = account_info['opening_date']
        
        # Get entity details
        entity_details = self.entity_map[entity_id]['details']
        
        # Generate transaction details
        transaction_type = random.choice([
            'Cash Deposit', 'Cash Withdrawal', 'Wire Transfer In', 'Wire Transfer Out',
            'Transfer', 'ACH Credit', 'ACH Debit'
        ])
        
        # Generate amount
        amount = random.uniform(500, 10000)
        
        # Generate timestamp - FIX DATE COMPARISON ISSUE
        one_year_ago = datetime.now() - timedelta(days=365)
        
        # Handle the type comparison correctly
        if hasattr(one_year_ago, 'date'):  # It's a datetime object
            # Convert account_open_date to datetime if it's a date
            if not hasattr(account_open_date, 'date'):  # It's a date object
                effective_start_date = datetime(
                    account_open_date.year,
                    account_open_date.month,
                    account_open_date.day,
                    0, 0, 0
                ) 
            else:
                effective_start_date = account_open_date
                
            # Compare and use the later date
            if effective_start_date > one_year_ago:
                start_date_for_fake = effective_start_date
            else:
                start_date_for_fake = one_year_ago
        else:
            # Fallback if datetime.now() somehow returns a date
            start_date_for_fake = one_year_ago
        
        timestamp = fake.date_time_between(
            start_date=start_date_for_fake,
            end_date=datetime.now()
        )
        
        # Rest of the function remains unchanged
        transaction = {
            'transaction_id': str(uuid.uuid4()),
            'account_id': account_id,
            'entity_id': entity_id,
            'transaction_type': transaction_type,
            'amount': round(amount, 2),
            'currency': 'USD',
            'timestamp': timestamp,
            'description': self._generate_transaction_description(transaction_type),
            'status': 'Completed',
            'counterparty_id': None,
            'counterparty_name': fake.company() if transaction_type in ['Wire Transfer In', 'Wire Transfer Out'] else None,
            'counterparty_bank': fake.company() + " Bank" if transaction_type in ['Wire Transfer In', 'Wire Transfer Out'] else None,
            'country': entity_details.get('country_code', 'US'),
            'ip_address': str(ipaddress.IPv4Address(random.randint(0, 2**32-1))),
            'created_at': timestamp,
            'updated_at': timestamp
        }
        
        # Create label record
        label = {
            'transaction_id': transaction['transaction_id'],
            'is_suspicious': 1,
            'TM-009': 1,  # Unusual Frequency
            'TM-006': 1,  # Account Velocity
        }
        
        # Set TM-001 if it's a large cash transaction
        if transaction_type in ['Cash Deposit', 'Cash Withdrawal'] and amount >= 10000:
            label['TM-001'] = 1
        else:
            label['TM-001'] = 0
            
        # Set TM-001a if it's an MSB account with cash transaction > 2500
        is_msb = (entity_type == 'business' and entity_details.get('is_msb', False))
        if transaction_type in ['Cash Deposit', 'Cash Withdrawal'] and is_msb and amount >= 2500:
            label['TM-001a'] = 1
        else:
            label['TM-001a'] = 0
        
        # Set all other flags to 0
        for rule in [
            'TM-002', 'TM-003', 'TM-004', 'TM-007', 'TM-008', 'TM-010',
            'TM-011', 'TM-012', 'TM-013', 'TM-014', 'TM-015', 'TM-018', 'TM-019', 'TM-020',
            'TM-021', 'TM-022', 'TM-025', 'TM-026', 'TM-029', 'TM-030', 'TM-031', 'TM-032', 'TM-033',
            'TM-035', 'TM-036', 'TM-037', 'TM-039', 'TM-040', 'TM-041'
        ]:
            label[rule] = 0
        
        return transaction, label
    
    def _generate_cross_border_transaction(self):
        """Generate a cross-border wire transaction (TM-031)"""
        account_id = random.choice(list(self.account_map.keys()))
        account_info = self.account_map[account_id]
        entity_id = account_info['entity_id']
        entity_type = account_info['entity_type']
        account_open_date = account_info['opening_date']
        
        # Get entity details
        entity_details = self.entity_map[entity_id]['details']
        
        # Generate transaction details
        transaction_type = random.choice([
            'Wire Transfer Out', 'Wire Transfer In', 
            'International Transfer Out', 'International Transfer In'
        ])
        
        # Generate amount
        amount = random.uniform(5000, 50000)
        
        # Generate timestamp
        timestamp = fake.date_time_between(
            start_date=self._safe_date_max(account_open_date, datetime.now() - timedelta(days=365)),
            end_date=datetime.now()
        )
        
        # Determine country (different from entity's country)
        entity_country = entity_details.get('country_code', 'US')
        countries = ['CA', 'GB', 'DE', 'FR', 'IT', 'ES', 'JP', 'CN', 'RU', 'AE', 'SG', 'HK']
        
        # Higher chance of selecting a high-risk country
        if random.random() < 0.3:
            country = random.choice(self.high_risk_countries)
        else:
            country = random.choice([c for c in countries if c != entity_country])
        
        # Determine if transaction involves unexpected location
        is_unexpected_location = random.random() < 0.6  # 60% chance
        
        # Create transaction record
        transaction = {
            'transaction_id': str(uuid.uuid4()),
            'account_id': account_id,
            'entity_id': entity_id,
            'transaction_type': transaction_type,
            'amount': round(amount, 2),
            'currency': random.choice(['USD', 'EUR', 'GBP']) if random.random() < 0.7 else random.choice(['JPY', 'CHF', 'AUD', 'CAD']),
            'timestamp': timestamp,
            'description': self._generate_transaction_description(transaction_type),
            'status': 'Completed',
            'counterparty_id': None,
            'counterparty_name': fake.company(),
            'counterparty_bank': fake.company() + " Bank",
            'country': country,
            'ip_address': str(ipaddress.IPv4Address(random.randint(0, 2**32-1))),
            'created_at': timestamp,
            'updated_at': timestamp
        }
        
        # Create label record
        label = {
            'transaction_id': transaction['transaction_id'],
            'is_suspicious': 1,
            'TM-031': 1,  # Cross-Border Wire
            'TM-033': 1 if is_unexpected_location else 0,  # Unexpected Location
        }
        
        # Set TM-004 if non-USD currency
        if transaction['currency'] != 'USD':
            label['TM-004'] = 1
        else:
            label['TM-004'] = 0
        
        # Set all other flags to 0
        for rule in [
            'TM-001', 'TM-001a', 'TM-002', 'TM-003', 'TM-006', 'TM-007', 'TM-008', 'TM-009', 'TM-010',
            'TM-011', 'TM-012', 'TM-013', 'TM-014', 'TM-015', 'TM-018', 'TM-019', 'TM-020',
            'TM-021', 'TM-022', 'TM-025', 'TM-026', 'TM-029', 'TM-030', 'TM-032',
            'TM-035', 'TM-036', 'TM-037', 'TM-039', 'TM-040', 'TM-041'
        ]:
            if rule != 'TM-004' and rule != 'TM-033':  # Skip if already set
                label[rule] = 0
        
        return transaction, label
    
    def _generate_ip_pattern_transaction(self):
        """Generate a transaction with suspicious IP pattern (TM-041)"""
        account_id = random.choice(list(self.account_map.keys()))
        account_info = self.account_map[account_id]
        entity_id = account_info['entity_id']
        entity_type = account_info['entity_type']
        account_open_date = account_info['opening_date']
        
        # Get entity details
        entity_details = self.entity_map[entity_id]['details']
        
        # Generate transaction details
        transaction_type = random.choice([
            'Online Payment', 'Transfer', 'Wire Transfer Out', 
            'ACH Credit', 'ACH Debit'
        ])
        
        # Generate amount
        amount = random.uniform(500, 10000)
        
        # Generate timestamp
        timestamp = fake.date_time_between(
            start_date=self._safe_date_max(account_open_date, datetime.now() - timedelta(days=365)),
            end_date=datetime.now()
        )
        
        # Generate suspicious IP
        # Either Tor exit node range, VPN range, or foreign country range
        ip_type = random.choice(['tor', 'vpn', 'foreign'])
        
        if ip_type == 'tor':
            # Simulate Tor exit node IP (not actual Tor IPs)
            ip_address = str(ipaddress.IPv4Address(random.randint(0xD0000000, 0xDFFFFFFF)))
        elif ip_type == 'vpn':
            # Simulate VPN IP (not actual VPN IPs)
            ip_address = str(ipaddress.IPv4Address(random.randint(0xA0000000, 0xAFFFFFFF)))
        else:
            # Foreign IP
            ip_address = str(ipaddress.IPv4Address(random.randint(0x50000000, 0x5FFFFFFF)))
        
        # Create transaction record
        transaction = {
            'transaction_id': str(uuid.uuid4()),
            'account_id': account_id,
            'entity_id': entity_id,
            'transaction_type': transaction_type,
            'amount': round(amount, 2),
            'currency': 'USD',
            'timestamp': timestamp,
            'description': self._generate_transaction_description(transaction_type),
            'status': 'Completed',
            'counterparty_id': None,
            'counterparty_name': fake.company() if transaction_type in ['Wire Transfer Out', 'ACH Credit', 'ACH Debit'] else None,
            'counterparty_bank': fake.company() + " Bank" if transaction_type in ['Wire Transfer Out', 'ACH Credit', 'ACH Debit'] else None,
            'country': entity_details.get('country_code', 'US'),
            'ip_address': ip_address,
            'created_at': timestamp,
            'updated_at': timestamp
        }
        
        # Create label record
        label = {
            'transaction_id': transaction['transaction_id'],
            'is_suspicious': 1,
            'TM-041': 1,  # IP Address Pattern
        }
        
        # Set TM-036 if it might involve anonymous payment methods
        if ip_type == 'tor' and random.random() < 0.7:
            label['TM-036'] = 1  # Anonymous Payment Methods
        else:
            label['TM-036'] = 0
        
        # Set all other flags to 0
        for rule in [
            'TM-001', 'TM-001a', 'TM-002', 'TM-003', 'TM-004', 'TM-006', 'TM-007', 'TM-008', 'TM-009', 'TM-010',
            'TM-011', 'TM-012', 'TM-013', 'TM-014', 'TM-015', 'TM-018', 'TM-019', 'TM-020',
            'TM-021', 'TM-022', 'TM-025', 'TM-026', 'TM-029', 'TM-030', 'TM-031', 'TM-032', 'TM-033',
            'TM-035', 'TM-037', 'TM-039', 'TM-040'
        ]:
            if rule != 'TM-036':  # Skip if already set
                label[rule] = 0
        
        return transaction, label
    
    def save_to_csv(self, output_dir):
        """
        Save generated data to CSV files.
        
        Args:
            output_dir (str): Directory to save CSV files
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Save each DataFrame to CSV
            for name, df in self.generated_data.items():
                file_path = os.path.join(output_dir, f"{name}.csv")
                df.to_csv(file_path, index=False)
                logger.info(f"Saved {len(df)} records to {file_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving data to CSV: {str(e)}")
            return False
    
    
    def generate_all_data(self, num_consumers=1000, num_businesses=200, 
                         num_transactions=50000, suspicious_pct=0.05):
        """
        Generate all required data for AML training.
        
        Args:
            num_consumers (int): Number of consumer records to generate
            num_businesses (int): Number of business records to generate
            num_transactions (int): Number of transactions to generate
            suspicious_pct (float): Percentage of transactions that should be suspicious
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("Starting data generation process")
            
            # Load schema information
            if not self.load_schema_information():
                logger.error("Failed to load schema information, cannot continue")
                return False
            
            # Generate entity data
            self.generate_consumer_data(num_consumers)
            self.generate_business_data(num_businesses)
            
            # Generate dependent data
            self.generate_dependent_data()
            
            # Generate beneficiary data
            self.generate_beneficiary_data()
            
            # Generate account data
            self.generate_account_data()
            
            # Generate transaction data
            self.generate_transaction_data(num_transactions, suspicious_pct)
            
            logger.info("Data generation complete")
            return True
            
        except Exception as e:
            logger.error(f"Error during data generation: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False


def prepare_data_for_ml(csv_dir, output_dir):
    """
    Prepare the generated data for machine learning.
    
    Args:
        csv_dir (str): Directory containing the CSV files
        output_dir (str): Directory to save prepared data
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test) DataFrames
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    logger.info("Preparing data for machine learning")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load transactions
    transactions_path = os.path.join(csv_dir, 'transactions.csv')
    transactions = pd.read_csv(transactions_path)
    
    # Load labels
    labels_path = os.path.join(csv_dir, 'aml_labels.csv')
    labels = pd.read_csv(labels_path)
    
    # Load entities
    customers_path = os.path.join(csv_dir, 'entity_customer.csv')
    businesses_path = os.path.join(csv_dir, 'entity_business.csv')
    
    if os.path.exists(customers_path):
        customers = pd.read_csv(customers_path)
    else:
        customers = pd.DataFrame()
        
    if os.path.exists(businesses_path):
        businesses = pd.DataFrame()
    
    # Load accounts
    accounts_path = os.path.join(csv_dir, 'account.csv')
    accounts = pd.read_csv(accounts_path)
    
    # Merge data
    # Start with transactions
    merged_data = transactions.copy()
    
    # Add account features
    account_features = [
        'account_id', 'account_type', 'account_status', 
        'opening_date', 'is_high_risk', 'days_inactive'
    ]
    if all(col in accounts.columns for col in account_features):
        accounts_slim = accounts[account_features]
        merged_data = merged_data.merge(accounts_slim, on='account_id', how='left')
    
    # Add entity features if available
    if not customers.empty:
        customer_features = [
            'entity_id', 'is_high_risk', 'risk_score', 'is_foreign',
            'country_code', 'is_pep', 'is_dormant'
        ]
        if all(col in customers.columns for col in customer_features):
            customers_slim = customers[customer_features]
            merged_data = merged_data.merge(customers_slim, on='entity_id', how='left', suffixes=('', '_customer'))
    
    if not businesses.empty:
        business_features = [
            'entity_id', 'is_high_risk', 'is_high_risk_industry', 'risk_score', 
            'is_foreign', 'country_code', 'is_msb', 'is_shell_company', 
            'is_cash_intensive'
        ]
        if all(col in businesses.columns for col in business_features):
            businesses_slim = businesses[business_features]
            merged_data = merged_data.merge(businesses_slim, on='entity_id', how='left', suffixes=('', '_business'))
    
    # Convert timestamp to datetime and extract features
    merged_data['timestamp'] = pd.to_datetime(merged_data['timestamp'], format='mixed', errors='coerce')
    merged_data['hour'] = merged_data['timestamp'].dt.hour
    merged_data['day_of_week'] = merged_data['timestamp'].dt.dayofweek
    merged_data['month'] = merged_data['timestamp'].dt.month
    merged_data['day'] = merged_data['timestamp'].dt.day
    merged_data['year'] = merged_data['timestamp'].dt.year
    
    # Flag transactions outside normal banking hours (8am-6pm)
    merged_data['outside_banking_hours'] = ((merged_data['hour'] < 8) | 
                                          (merged_data['hour'] > 18)).astype(int)
    
    # Flag weekend transactions
    merged_data['is_weekend'] = (merged_data['day_of_week'] > 4).astype(int)
    
    # Merge with labels
    final_data = merged_data.merge(labels, on='transaction_id', how='left')
    
    # Handle missing values
    # For numeric columns, fill with 0
    numeric_cols = final_data.select_dtypes(include=['number']).columns
    final_data[numeric_cols] = final_data[numeric_cols].fillna(0)
    
    # For categorical columns, fill with 'Unknown'
    categorical_cols = final_data.select_dtypes(include=['object']).columns
    final_data[categorical_cols] = final_data[categorical_cols].fillna('Unknown')
    
    # Prepare features and target
    # Remove columns that should not be used as features
    drop_cols = [
        'transaction_id', 'account_id', 'entity_id', 'timestamp', 
        'created_at', 'updated_at', 'description', 'counterparty_id', 
        'counterparty_name', 'counterparty_bank'
    ]
    
    # Remove target variables from features
    target_cols = ['is_suspicious'] + [col for col in final_data.columns if col.startswith(('TM-', 'TM_'))]
    
    # Create feature matrix
    X = final_data.drop(columns=drop_cols + target_cols, errors='ignore')
    
    # Create target vector
    y = final_data['is_suspicious']
    
    # Create specific rule targets
    rule_targets = final_data[[col for col in final_data.columns if col.startswith(('TM-', 'TM_'))]]
    
    # Convert categorical features to numeric
    # Get list of categorical columns remaining in X
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    # One-hot encode categorical columns
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Save prepared data
    X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
    
    # Save rule targets separately
    rule_train = rule_targets.iloc[X_train.index]
    rule_test = rule_targets.iloc[X_test.index]
    rule_train.to_csv(os.path.join(output_dir, 'rule_targets_train.csv'), index=False)
    rule_test.to_csv(os.path.join(output_dir, 'rule_targets_test.csv'), index=False)
    
    # Save merged data for reference
    final_data.to_csv(os.path.join(output_dir, 'merged_data.csv'), index=False)
    
    logger.info(f"Data preparation complete. Files saved to {output_dir}")
    logger.info(f"Training set size: {len(X_train)} rows, {len(X_train.columns)} features")
    logger.info(f"Testing set size: {len(X_test)} rows, {len(X_test.columns)} features")
    
    return X_train, X_test, y_train, y_test


def main():
    """Main function to run the data generator"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate synthetic AML training data')
    parser.add_argument('--config', type=str, default='aml_config.properties', help='Path to config file')
    parser.add_argument('--output_dir', type=str, default='aml_data', help='Directory to save output files')
    parser.add_argument('--num_consumers', type=int, default=1000, help='Number of consumer records to generate')
    parser.add_argument('--num_businesses', type=int, default=200, help='Number of business records to generate')
    parser.add_argument('--num_transactions', type=int, default=50000, help='Number of transactions to generate')
    parser.add_argument('--suspicious_pct', type=float, default=0.05, help='Percentage of suspicious transactions')
    parser.add_argument('--save_csv', action='store_true', help='Save data to CSV files')
    parser.add_argument('--prepare_ml', action='store_true', help='Prepare data for machine learning')
    
    args = parser.parse_args()
    
    # Initialize the data generator
    generator = AMLTrainingDataGenerator(args.config)
    
    # Generate data
    if generator.generate_all_data(
        num_consumers=args.num_consumers,
        num_businesses=args.num_businesses,
        num_transactions=args.num_transactions,
        suspicious_pct=args.suspicious_pct
    ):
        logger.info("Data generation successful")
        
        # Save to CSV files
        if args.save_csv:
            if generator.save_to_csv(args.output_dir):
                logger.info(f"Successfully saved data to CSV files in {args.output_dir}")
            else:
                logger.error("Failed to save data to CSV files")
        
        
        # Prepare data for machine learning
        if args.prepare_ml and args.save_csv:
            ml_output_dir = os.path.join(args.output_dir, 'ml_data')
            prepare_data_for_ml(args.output_dir, ml_output_dir)
            logger.info(f"Successfully prepared data for machine learning in {ml_output_dir}")
    else:
        logger.error("Data generation failed")


if __name__ == "__main__":
    main()