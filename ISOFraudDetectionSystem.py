import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from process_files import MultiFormatProcessor
from return_processor import ACHReturnProcessor, ReturnProcessorIntegration
from mule_account_detector import MuleAccountDetector
from account_takeover_detector import AccountTakeoverDetector
from dataclasses import asdict, is_dataclass
from app_fraud_detector import APPFraudDetector
from platformlist_manager import PlatformListManager, ListType
from transaction_logger import TransactionLogger
import sys
import json
import os
from xgboost_dll_resolver import setup_xgboost_dll
# Get the directory where the executable is located
# Construct the path to the xgboost.dll (assuming it's in the same directory)
# dll_path_str="C:\\Users\\kisho\\OneDrive - BlinkAIPayments\\fraud\\models\\9Feb2025\\ver3.2\\"
# dll_path = os.path.join(os.getcwd(), "xgboost.dll")  # Or "libxgboost.dll" depending on your XGBoost version
# dll_path = os.path.join(dll_path_str, "xgboost.dll")

# Set the environment variable. This is the most reliable way to tell XGBoost where to find its library.
# os.environ['XGBOOST_DLL'] = dll_path
# if os.path.exists(dll_path):
#     os.environ['XGBOOST_DLL'] = dll_path
#     print(f"XGBoost DLL path set to: {dll_path}")
# else:
#     print(f"ERROR: xgboost.dll not found at expected path: {dll_path}")
#     sys.exit(1)  # Or handle the error as needed
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Union, Set
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta, timezone  
import pickle
import tqdm
from fuzzywuzzy import fuzz
from enum import Enum
import xml.etree.ElementTree as ET
import csv
from pathlib import Path
from typing import List, Optional
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from db_utils import DatabaseConnection
# from pattern_discovery import PatternDiscoveryIntegration
setup_xgboost_dll()
import xgboost as xgb
from csv_utils import CSVDatabaseConnection
from configreader import ConfigReader
import re
from transaction_history import TransactionHistoryLoader
from typing import Dict, List, Optional, Union, Any
import xml.etree.ElementTree as ET
import logging
from enum import Enum
from datetime_utils import normalize_datetime, normalize_dataframe_dates, filter_by_timespan
import tempfile
import decimal
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PaymentChannel(Enum):
    """Supported payment channels"""
    ZELLE = "ZELLE"
    SWIFT = "SWIFT"
    FEDNOW = "FEDNOW"
    ACH = "ACH"
    WIRE = "WIRE"

class DatetimeHandler:
    @staticmethod
    def standardize_datetime(df, columns):
        """
        Standardize datetime columns to a consistent format
        
        Args:
            df (pd.DataFrame): Input DataFrame
            columns (list): Columns to standardize
        
        Returns:
            pd.DataFrame: DataFrame with standardized datetime columns
        """
        for col in columns:
            try:
                # Convert to pandas Timestamp with UTC timezone
                df[col] = pd.to_datetime(df[col], utc=True)
                
                # Ensure consistent datetime representation
                df[col] = df[col].dt.tz_convert('UTC')
            except Exception as e:
                print(f"Error converting column {col}: {e}")
                # Fallback: try alternative conversion methods
                df[col] = df[col].apply(lambda x: DatetimeHandler._safe_datetime_convert(x))
        
        return df
    
    @staticmethod
    def _safe_datetime_convert(value):
        """
        Safe datetime conversion method
        
        Args:
            value: Input value to convert
        
        Returns:
            Standardized datetime or original value if conversion fails
        """
        try:
            # Multiple conversion attempts
            if isinstance(value, pd.Timestamp):
                return value.tz_convert('UTC')
            
            if isinstance(value, datetime):
                return pd.Timestamp(value).tz_localize('UTC')
            
            if isinstance(value, np.datetime64):
                return pd.Timestamp(value).tz_localize('UTC')
            
            # Try parsing string representations
            if isinstance(value, str):
                try:
                    # Try parsing with multiple formats
                    formats = [
                        '%Y-%m-%d %H:%M:%S%z',
                        '%Y-%m-%d %H:%M:%S',
                        '%Y-%m-%dT%H:%M:%S',
                        '%Y-%m-%d'
                    ]
                    
                    for fmt in formats:
                        try:
                            parsed_dt = datetime.strptime(value, fmt)
                            return pd.Timestamp(parsed_dt).tz_localize('UTC')
                        except ValueError:
                            continue
                except Exception:
                    pass
            
            # If all else fails, use current time
            return pd.Timestamp.now(tz='UTC')
        
        except Exception as e:
            print(f"Conversion error: {e}")
            return pd.Timestamp.now(tz='UTC')

def convert_timestamp_to_datetime(timestamp):
    """
    Convert various timestamp formats to a consistent datetime
    
    Args:
        timestamp: Input timestamp
    
    Returns:
        datetime object in UTC
    """
    try:
        # Handle pandas Timestamp
        if isinstance(timestamp, pd.Timestamp):
            return timestamp.to_pydatetime().astimezone(timezone.utc)
        
        # Handle datetime object
        if isinstance(timestamp, datetime):
            # Ensure it's in UTC
            return timestamp.astimezone(timezone.utc) if timestamp.tzinfo else timestamp.replace(tzinfo=timezone.utc)
        
        # Handle numpy datetime64
        if isinstance(timestamp, np.datetime64):
            return pd.Timestamp(timestamp).to_pydatetime().astimezone(timezone.utc)
        
        # Handle string representations
        if isinstance(timestamp, str):
            try:
                # Try parsing with multiple formats
                formats = [
                    '%Y-%m-%d %H:%M:%S%z',
                    '%Y-%m-%d %H:%M:%S',
                    '%Y-%m-%dT%H:%M:%S',
                    '%Y-%m-%d'
                ]
                
                for fmt in formats:
                    try:
                        parsed_dt = datetime.strptime(timestamp, fmt)
                        return parsed_dt.replace(tzinfo=timezone.utc)
                    except ValueError:
                        continue
            except Exception:
                pass
        
        # If all else fails, return current time in UTC
        return datetime.now(timezone.utc)
    
    except Exception as e:
        print(f"Timestamp conversion error: {e}")
        return datetime.now(timezone.utc)

def ensure_utc_timezone(dt):
    """
    Ensure datetime is in UTC timezone
    
    Args:
        dt: Input datetime
    
    Returns:
        datetime in UTC timezone
    """
    try:
        if dt is None:
            return datetime.now(timezone.utc)
        
        # If no timezone info, assume UTC
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        
        # Convert to UTC
        return dt.astimezone(timezone.utc)
    
    except Exception as e:
        print(f"Timezone conversion error: {e}")
        return datetime.now(timezone.utc)

class ISO20022MessageType(Enum):
    """Supported ISO20022 message types"""
    # Payment Initiation
    PAIN_001 = "pain.001.001.09"  # CustomerCreditTransferInitiation
    PAIN_002 = "pain.002.001.09"  # CustomerPaymentStatusReport
    
    # Payment Clearing and Settlement
    PACS_008 = "pacs.008.001.08"  # FIToFICustomerCreditTransfer
    PACS_002 = "pacs.002.001.10"  # FIToFIPaymentStatusReport
    PACS_004 = "pacs.004.001.09"  # PaymentReturn
    
    # Cash Management
    CAMT_052 = "camt.052.001.08"  # BankToCustomerAccountReport
    CAMT_053 = "camt.053.001.08"  # BankToCustomerStatement
    CAMT_054 = "camt.054.001.08"  # BankToCustomerDebitCreditNotification

@dataclass
class ISO20022Header:
    """Common ISO20022 message header elements"""
    message_id: str
    creation_datetime: datetime
    number_of_transactions: int
    control_sum: Optional[float]
    initiating_party: Dict[str, str]
    message_type: ISO20022MessageType

@dataclass
class ISO20022Transaction:
    """Common transaction elements across message types"""
    transaction_id: str
    end_to_end_id: str
    amount: float
    currency: str
    debtor: Dict[str, str]
    creditor: Dict[str, str]
    remittance_info: Optional[str]
    purpose: Optional[str]
    additional_info: Dict[str, str]


class TransactionMetrics:
    def __init__(self, 
                 # Existing parameters
                 velocity_24h=0, amount_24h=0.0, unique_recipients_24h=0,
                 velocity_7d=0, amount_7d=0.0, unique_recipients_7d=0,
                 avg_amount_30d=0.0, std_amount_30d=0.0, new_recipient=True,
                 cross_border=False, high_risk_country=False,route_count_1h=0,
                 amount_30d=0.0,avg_amount_24h=0.0,avg_amount_7d=0.0,tx_count_1h=0,
                 account_age_days=0, activity_days_before_transaction=0,
                 days_since_first_transaction=0, days_since_last_transaction=0,
                 weekday=0, hour_of_day=12, is_business_hours=True,
                 typical_transaction_hour_flag=False, typical_transaction_weekday_flag=False,
                 rounded_amount_ratio=0.0, diverse_sources_count=0,
                 funnel_pattern_ratio=0.0, off_hours_transaction_ratio=0.0,
                 rapid_withdrawal_flag=False, account_emptying_percentage=0.0,
                 velocity_ratio=1.0, amount_zscore=0.0, seasonal_adjustment_factor=1.0,
                 seasonal_pattern_deviation=0.0, quarterly_activity_comparison=0.0,
                 year_over_year_behavior_change=0.0, new_beneficiary_flag=False,
                 beneficiary_risk_score=0.0, recipient_pattern_change_flag=False,
                 days_since_first_payment_to_recipient=999, dormant_recipient_reactivation=False, 
                 recipient_concentration_ratio=0.0, total_unique_recipients_365d=0,
                 recipient_network_density=0.0, new_network_connections=0,
                 source_target_previous_transfers=0, transaction_pattern_break_risk_score=0.0,
                 amount_anomaly_risk_score=0.0, new_recipient_risk_score=0.0,
                 round_amount_risk_score=0.0, unusual_timing_risk_score=0.0,
                 after_hours_risk_score=0.0, unusual_location_risk_score=0.0,
                 velocity_risk_score=0.0, is_round_amount=False, app_fraud_risk=0.0,
                 known_scam_recipient_flag=False, threshold_days_since_last_recipient=999,
                 unusual_location_flag=False, geo_velocity_impossible_flag=False,
                 distance_from_common_locations=0.0, 
                 mule_layering_pattern=False,
                 annual_velocity_percentile=0.0,
                 seasonality_pattern_deviation=0.0,
                 mule_rapid_disbursement=False, mule_new_account_high_volume=False,
                 mule_geographic_diversity=0, 
                 new_recipient_large_amount=False,
                 high_risk_corridor_flag=False, collect_transfer_pattern_flag=False,
                 source_account_risk_score=0.0, target_account_risk_score=0.0,
                 annual_amount_trend=0.0, payment_growth_rate=0.0,
                 cyclical_amount_pattern_break=False, amount_volatility_365d=0.0,
                 account_milestone_deviation=0.0, major_behavior_shift_timestamps=None,
                 account_maturity_risk_score=0.0, lifecycle_stage_appropriate_behavior=True,
                 behavioral_consistency_score=0.0, pattern_break_frequency_365d=0,
                 transaction_predictability_score=0.0, rolling_risk_trend_365d=0.0,
                 behavioral_change_velocity=0.0, risk_volatility_365d=0.0,
                 historical_fraud_attempts=0, aggregated_risk_score=0.0,
                 combined_risk=0.0, channel_adjusted_score=0.0, recipient_risk=0.0,
                 behavioral_score=0.0, confidence_level="LOW", multiple_beneficiaries_risk_score=0.0,
                 new_beneficiary_risk_score=0.0, amount_pattern_risk_score=0.0,
                 unusual_access_time_flag=False, unusual_time_risk_score=0.0,
                 triggered_indicators_count=0, is_suspected_app_fraud=False,
                 is_suspected_takeover=False, is_account_takeover=False,
                 multi_indicator_present=False, avg_velocity_30d=0.0,
                 amount_30d_avg=0.0, amount_30d_std=0.0, periodic_payment_disruption=False,
                 avg_monthly_transaction_count=0.0, avg_monthly_transaction_volume=0.0,
                 transaction_frequency_stddev=0.0, recipient_inactivity_period=0,
                 recipient_frequency_distribution=None, recipient_transaction_history_365d=None,
                 login_frequency_30d=0.0, device_change_recency=999, credential_age_days=999,
                 failed_login_attempts_24h=0, password_reset_recency=999,
                 mfa_bypass_attempts=0, mfa_status=False, is_known_device=True,
                 device_reputation_score=0.0, browser_fingerprint_match=True,
                 session_duration=0, session_anomaly_score=0.0, ip_address_change_flag=False,
                 vpn_proxy_flag=False):
        
        # Existing attributes
        self.velocity_24h = velocity_24h
        self.amount_24h = amount_24h
        self.unique_recipients_24h = unique_recipients_24h
        self.velocity_7d = velocity_7d
        self.amount_7d = amount_7d
        self.unique_recipients_7d = unique_recipients_7d
        self.avg_amount_30d = avg_amount_30d
        self.std_amount_30d = std_amount_30d
        self.new_recipient = new_recipient
        self.cross_border = cross_border
        self.high_risk_country = high_risk_country
        
        # New attributes - account-level
        self.account_age_days = account_age_days
        self.activity_days_before_transaction = activity_days_before_transaction
        self.days_since_first_transaction = days_since_first_transaction
        self.days_since_last_transaction = days_since_last_transaction
        
        # Temporal behavior
        self.is_business_hours = is_business_hours
        self.weekday = weekday
        self.hour_of_day = hour_of_day
        self.typical_transaction_hour_flag = typical_transaction_hour_flag
        self.typical_transaction_weekday_flag = typical_transaction_weekday_flag
        
        # Transaction patterns
        self.rounded_amount_ratio = rounded_amount_ratio
        self.diverse_sources_count = diverse_sources_count
        self.funnel_pattern_ratio = funnel_pattern_ratio
        self.off_hours_transaction_ratio = off_hours_transaction_ratio
        self.rapid_withdrawal_flag = rapid_withdrawal_flag
        self.account_emptying_percentage = account_emptying_percentage
        self.velocity_ratio = velocity_ratio
        self.amount_zscore = amount_zscore
        
        # Seasonal patterns
        self.seasonal_adjustment_factor = seasonal_adjustment_factor
        self.seasonal_pattern_deviation = seasonal_pattern_deviation
        self.quarterly_activity_comparison = quarterly_activity_comparison
        self.year_over_year_behavior_change = year_over_year_behavior_change
        self.periodic_payment_disruption = periodic_payment_disruption
        
        # Beneficiary features
        self.new_beneficiary_flag = new_beneficiary_flag
        self.beneficiary_risk_score = beneficiary_risk_score
        self.recipient_pattern_change_flag = recipient_pattern_change_flag
        self.days_since_first_payment_to_recipient = days_since_first_payment_to_recipient
        self.dormant_recipient_reactivation = dormant_recipient_reactivation
        self.recipient_concentration_ratio = recipient_concentration_ratio
        self.new_recipient_large_amount = new_recipient_large_amount
        
        # Network metrics
        self.total_unique_recipients_365d = total_unique_recipients_365d
        self.recipient_network_density = recipient_network_density
        self.new_network_connections = new_network_connections
        self.source_target_previous_transfers = source_target_previous_transfers
        
        # Risk scores
        self.transaction_pattern_break_risk_score = transaction_pattern_break_risk_score
        self.amount_anomaly_risk_score = amount_anomaly_risk_score
        self.new_recipient_risk_score = new_recipient_risk_score
        self.round_amount_risk_score = round_amount_risk_score
        self.unusual_timing_risk_score = unusual_timing_risk_score
        self.after_hours_risk_score = after_hours_risk_score
        self.unusual_location_risk_score = unusual_location_risk_score
        self.velocity_risk_score = velocity_risk_score
        self.unusual_time_risk_score = unusual_time_risk_score
        self.multiple_beneficiaries_risk_score = multiple_beneficiaries_risk_score
        self.new_beneficiary_risk_score = new_beneficiary_risk_score
        self.amount_pattern_risk_score = amount_pattern_risk_score

        self.avg_monthly_transaction_volume = avg_monthly_transaction_volume
        self.transaction_frequency_stddev = transaction_frequency_stddev
        self.avg_monthly_transaction_count = avg_monthly_transaction_count 
        
        # Additional fraud indicators
        self.is_round_amount = is_round_amount
        self.app_fraud_risk = app_fraud_risk
        self.known_scam_recipient_flag = known_scam_recipient_flag
        self.threshold_days_since_last_recipient = threshold_days_since_last_recipient
        
        # Location features
        self.unusual_location_flag = unusual_location_flag
        self.geo_velocity_impossible_flag = geo_velocity_impossible_flag
        self.distance_from_common_locations = distance_from_common_locations
        self.vpn_proxy_flag = vpn_proxy_flag
        self.ip_address_change_flag = ip_address_change_flag
        
        # Mule detection features
        self.mule_layering_pattern = mule_layering_pattern
        self.mule_rapid_disbursement = mule_rapid_disbursement
        self.mule_new_account_high_volume = mule_new_account_high_volume
        self.mule_geographic_diversity = mule_geographic_diversity
        self.high_risk_corridor_flag = high_risk_corridor_flag
        self.collect_transfer_pattern_flag = collect_transfer_pattern_flag
        self.source_account_risk_score = source_account_risk_score
        self.target_account_risk_score = target_account_risk_score
        
        # Long-term pattern analysis
        self.annual_amount_trend = annual_amount_trend
        self.payment_growth_rate = payment_growth_rate
        self.cyclical_amount_pattern_break = cyclical_amount_pattern_break
        self.amount_volatility_365d = amount_volatility_365d
        self.account_milestone_deviation = account_milestone_deviation
        self.major_behavior_shift_timestamps = major_behavior_shift_timestamps if major_behavior_shift_timestamps else []
        self.account_maturity_risk_score = account_maturity_risk_score
        self.lifecycle_stage_appropriate_behavior = lifecycle_stage_appropriate_behavior
        
        # Behavioral consistency metrics
        self.behavioral_consistency_score = behavioral_consistency_score
        self.pattern_break_frequency_365d = pattern_break_frequency_365d
        self.transaction_predictability_score = transaction_predictability_score
        
        # Risk trending
        self.rolling_risk_trend_365d = rolling_risk_trend_365d
        self.behavioral_change_velocity = behavioral_change_velocity
        self.risk_volatility_365d = risk_volatility_365d
        self.historical_fraud_attempts = historical_fraud_attempts
        
        # Aggregated risk metrics
        self.aggregated_risk_score = aggregated_risk_score
        self.combined_risk = combined_risk
        self.channel_adjusted_score = channel_adjusted_score
        self.recipient_risk = recipient_risk
        self.behavioral_score = behavioral_score
        self.confidence_level = confidence_level
        
        # Authentication and device metrics
        self.login_frequency_30d = login_frequency_30d
        self.device_change_recency = device_change_recency
        self.credential_age_days = credential_age_days
        self.failed_login_attempts_24h = failed_login_attempts_24h
        self.password_reset_recency = password_reset_recency
        self.mfa_bypass_attempts = mfa_bypass_attempts
        self.mfa_status = mfa_status
        self.is_known_device = is_known_device
        self.device_reputation_score = device_reputation_score
        self.browser_fingerprint_match = browser_fingerprint_match
        self.session_duration = session_duration
        self.session_anomaly_score = session_anomaly_score
        
        # Flag indicators
        self.unusual_access_time_flag = unusual_access_time_flag
        self.triggered_indicators_count = triggered_indicators_count
        self.is_suspected_app_fraud = is_suspected_app_fraud
        self.is_suspected_takeover = is_suspected_takeover
        self.is_account_takeover = is_account_takeover
        self.multi_indicator_present = multi_indicator_present
        
        # Aliases for backward compatibility
        self.amount_30d_avg = avg_amount_30d
        self.amount_30d_std = std_amount_30d
        self.avg_velocity_30d = velocity_24h / 24.0 if velocity_24h else 0
        
        # Recipient analysis
        self.recipient_inactivity_period = recipient_inactivity_period
        self.recipient_frequency_distribution = recipient_frequency_distribution if recipient_frequency_distribution else {}
        self.recipient_transaction_history_365d = recipient_transaction_history_365d if recipient_transaction_history_365d else []

class MessageChannelMapper:
    """Maps between ISO20022 message types and payment channels"""
    
    def __init__(self):
        # Default mapping from message type to channel
        self.message_to_channel = {
            ISO20022MessageType.PAIN_001: {
                # "default": PaymentChannel.ACH, - this is being processed with incoming raw ACH files, - 13-Mar-2025
                "default" : PaymentChannel.WIRE,
                "rules": [
                    (lambda msg: "ZELLE" in str(msg), PaymentChannel.ZELLE),
                    (lambda msg: "FedNow" in str(msg), PaymentChannel.FEDNOW)
                ]
            },
            ISO20022MessageType.PACS_008: {
                "default": PaymentChannel.SWIFT,
                "rules": [
                    (lambda msg: "FedNow" in str(msg), PaymentChannel.FEDNOW)
                ]
            },
            ISO20022MessageType.CAMT_054: {
                "default": PaymentChannel.WIRE,
                "rules": []
            }
        }
        
        # Mapping from channel to preferred message type
        self.channel_to_message = {
            PaymentChannel.ZELLE: ISO20022MessageType.PAIN_001,
            PaymentChannel.SWIFT: ISO20022MessageType.PACS_008,
            PaymentChannel.FEDNOW: ISO20022MessageType.PACS_008,
            # PaymentChannel.ACH: ISO20022MessageType.PAIN_001, #Processed using raw ACH files, no longer being verified via ISO2022 - 13-Mar-2025
            PaymentChannel.WIRE: ISO20022MessageType.PACS_008
        }

    def determine_channel(self, message_type: ISO20022MessageType, 
                         message_content: Optional[str] = None) -> PaymentChannel:
        """Determine payment channel from message type and content"""
        if message_type not in self.message_to_channel:
            raise ValueError(f"Unsupported message type: {message_type}")
            
        mapping = self.message_to_channel[message_type]
        
        # Check content-specific rules if content is provided
        if message_content:
            for rule, channel in mapping["rules"]:
                if rule(message_content):
                    return channel
                    
        return mapping["default"]

    def get_message_type(self, channel: PaymentChannel) -> ISO20022MessageType:
        """Get preferred message type for a payment channel"""
        return self.channel_to_message[channel]

class ChannelAlertDetector:
    """Detects unusual patterns specific to each payment channel"""
    
    def __init__(self, config_reader: ConfigReader, db_connection=None):
        self.config_reader = config_reader
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        # Store the database connection
        self.db = db_connection
        self.channel_thresholds = self._load_channel_thresholds()
        self.high_risk_return_codes = self._load_config_list('ach.high_risk_return_codes', 
                                                            ['R01', 'R02', 'R03', 'R04', 'R05', 'R08', 'R20'])
        
        # Load ACH off-hours configuration
        self.ach_off_hours_start = int(self.config_reader.get_property('ach.off_hours_start', '20'))
        self.ach_off_hours_end = int(self.config_reader.get_property('ach.off_hours_end', '6'))
        self.ach_off_hours_severity = self.config_reader.get_property('ach.off_hours_severity', 'MEDIUM')
        
    def __getattr__(self, name):
        """Handle attributes that aren't explicitly defined"""
        if name in ['suspicious_odfi_list', 'suspicious_rdfi_list']:
            # Return empty list for these attributes if they haven't been set elsewhere
            return []
        # Raise normal AttributeError for other missing attributes
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def _load_config_list(self, config_key, default_list=None):
        """
        Load a list from configuration
        
        :param config_key: Configuration key
        :param default_list: Default list if not found
        :return: List of values
        """
        if default_list is None:
            default_list = []
            
        config_value = self.config_reader.get_property(config_key, '')
        if not config_value:
            return default_list
            
        # Split by commas and trim whitespace
        return [item.strip() for item in config_value.split(',')]
        
    def _load_channel_thresholds(self) -> Dict[str, Dict]:
        """Load channel-specific thresholds from config"""
        try:
            channel_defaults = {
                'min_amount': 0.0,
                'max_amount': float('inf'),
                'high_amount': 10000.0,
                'velocity_24h': 10,
                'business_hours_start': 9,
                'business_hours_end': 17,
                'weekend_allowed': False
            }
            
            return {
                'WIRE': self._load_channel_config('wire', channel_defaults),
                'SWIFT': self._load_channel_config('swift', channel_defaults),
                'FEDNOW': self._load_channel_config('fednow', channel_defaults),
                'ZELLE': self._load_channel_config('zelle', channel_defaults),
                'ACH': self._load_channel_config('ach', channel_defaults)
            }
        except Exception as e:
            self.logger.error(f"Error loading channel thresholds: {str(e)}")
            return {}
        
    def _load_channel_config(self, channel: str, defaults: Dict) -> Dict:
        """Load configuration for specific channel with defaults"""
        try:
            config= {
                'min_amount': float(self.config_reader.get_property(f'{channel}.threshold.min_amount', defaults['min_amount'])),
                'max_amount': float(self.config_reader.get_property(f'{channel}.threshold.max_amount', defaults['max_amount'])),
                'high_amount': float(self.config_reader.get_property(f'{channel}.threshold.high_amount', defaults['high_amount'])),
                'velocity_24h': int(self.config_reader.get_property(f'{channel}.threshold.velocity_24h', defaults['velocity_24h'])),
                'business_hours_start': int(self.config_reader.get_property(f'{channel}.business_hours.start', defaults['business_hours_start'])),
                'business_hours_end': int(self.config_reader.get_property(f'{channel}.business_hours.end', defaults['business_hours_end'])),
                'weekend_allowed': bool(self.config_reader.get_property(f'{channel}.allow_weekend', 'false') == 'true')
            }

            # Add specific properties if this is a channel
            if channel == 'fednow':
                config.update({
                    'rapid_succession_count': int(self.config_reader.get_property(f'{channel}.threshold.rapid_succession_count', '3')),
                    'rapid_succession_interval': int(self.config_reader.get_property(f'{channel}.threshold.rapid_succession_interval', '30'))
                })
            
            if channel == 'swift':
                config.update({
                    'rapid_succession_count': int(self.config_reader.get_property(f'{channel}.threshold.rapid_succession_count', '5')),
                    'rapid_succession_interval': int(self.config_reader.get_property(f'{channel}.threshold.rapid_succession_interval', '30'))
                })
            
            if channel == 'zelle':
                config.update({
                    'rapid_succession_count': int(self.config_reader.get_property(f'{channel}.threshold.rapid_succession_count', '5')),
                    'rapid_succession_interval': int(self.config_reader.get_property(f'{channel}.threshold.rapid_succession_interval', '30'))
                })

            if channel == 'wire':
                config.update({
                    'rapid_succession_count': int(self.config_reader.get_property(f'{channel}.threshold.rapid_succession_count', '5')),
                    'rapid_succession_interval': int(self.config_reader.get_property(f'{channel}.threshold.rapid_succession_interval', '30'))
                })

            if channel == 'ach':
                config.update({
                    'rapid_succession_count': int(self.config_reader.get_property(f'{channel}.threshold.rapid_succession_count', '5')),
                    'rapid_succession_interval': int(self.config_reader.get_property(f'{channel}.threshold.rapid_succession_interval', '30'))
                })

            return config
        
        except Exception as e:
            self.logger.error(f"Error loading config for channel {channel}: {str(e)}")
            return dict(defaults)
        
    def _calculate_rdfi_risk(self, rdfi_id: str) -> float:
        """
        Calculate risk score for RDFI ID based on suspicious patterns using PlatformListManager
        
        :param rdfi_id: Receiving Depository Financial Institution ID
        :return: Risk score (0-100) indicating risk level
        """
        base_score = 0
        
        # Get risk score config values with defaults
        blacklist_score = float(self.config_reader.get_property('rdfi.risk.blacklist_score', '100'))
        format_error_score = float(self.config_reader.get_property('rdfi.risk.format_error_score', '70'))
        invalid_district_score = float(self.config_reader.get_property('rdfi.risk.invalid_district_score', '80'))
        targeted_list_score = float(self.config_reader.get_property('rdfi.risk.targeted_list_score', '90'))
        
        # Initialize database connection if it doesn't exist
        if not hasattr(self, 'db') or self.db is None:
            try:
                self.db = DatabaseConnection(self.config_reader.get_property('config.path', './config.properties'))
                self.logger.info("Database connection initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize database connection: {str(e)}")
                return base_score

        # Initialize platform list manager if not already done
        try:
            # Check if the db connection is valid before using it
            if self.db and hasattr(self.db, 'reconnect'):
                self.db.reconnect()
                
                if not hasattr(self, 'platform_list_manager'):
                    redis_config = {
                        "host": self.config_reader.get_property('redis.host', 'localhost'),
                        "port": int(self.config_reader.get_property('redis.port', '6379')),
                        "db": int(self.config_reader.get_property('redis.db', '0'))
                    }
                    
                    self.logger.info("Initializing PlatformListManager...")
                    self.platform_list_manager = PlatformListManager(self.db, redis_config)
                    self.logger.info("PlatformListManager initialized successfully")
            else:
                self.logger.error("Database connection is not properly initialized")
                return base_score
        except Exception as e:
            self.logger.error(f"Failed to initialize PlatformListManager: {str(e)}")
            self.platform_list_manager = None
            return base_score

        
        # Search for the RDFI in the platform lists
        if self.platform_list_manager:
            try:
                list_type, entry = self.platform_list_manager.search_entry("bank_routing_number", rdfi_id)
                
                # Process the result if found
                if list_type and entry:
                    # Check list type
                    if list_type == ListType.BLACKLIST:
                        return blacklist_score
                        
                    if list_type == ListType.RDFI_SUSPICIOUS:
                        return blacklist_score
                        
                    # Check for identifiers or notes with "targeted"
                    if entry.get('notes') and 'targeted' in str(entry.get('notes', '')).lower():
                        base_score = max(base_score, targeted_list_score)
                        
                    # Use risk score if available
                    if entry.get('risk_score') is not None:
                        base_score = max(base_score, float(entry.get('risk_score', 0)))
            except Exception as e:
                self.logger.error(f"Error searching for RDFI {rdfi_id}: {str(e)}")
                
        
        # Check for patterns indicating suspicious activity (fallback checks)
        # Non-standard routing number formats
        if not rdfi_id.isdigit() or len(rdfi_id) < 8:
            base_score = max(base_score, format_error_score)
            
        # Check Federal Reserve district (first digit)
        # 0 is not a valid district in the US
        if rdfi_id.startswith('0'):
            base_score = max(base_score, invalid_district_score)
        
        return base_score

    def _is_suspicious_rdfi(self, rdfi_id: str) -> bool:
        """
        Legacy method for backward compatibility.
        Converts risk score to boolean.
        
        :param rdfi_id: Receiving Depository Financial Institution ID
        :return: Boolean indicating if suspicious
        """
        risk_threshold = float(self.config_reader.get_property('rdfi.risk.suspicious_threshold', '50'))
        risk_score = self._calculate_rdfi_risk(rdfi_id)
        return risk_score >= risk_threshold


    def _is_suspicious_odfi(self, odfi_id: str) -> float:
        """
        Check if ODFI ID matches suspicious patterns using PlatformListManager
        
        :param odfi_id: Originating Depository Financial Institution ID
        :return: Risk score for the ODFI
        """
        base_score = 0
        # Get risk score config values with defaults
        blacklist_score = float(self.config_reader.get_property('odfi.risk.blacklist_score', '100'))
        format_error_score = float(self.config_reader.get_property('odfi.risk.format_error_score', '70'))
        new_odfi_score = float(self.config_reader.get_property('odfi.risk.new_odfi_score', '60'))
        high_volume_off_hours_score = float(self.config_reader.get_property('odfi.risk.high_volume_off_hours', '85'))
        
        # Initialize platform list manager if not already done
        try:
            self.db.reconnect()
            if not hasattr(self, 'platform_list_manager'):
                
                redis_config = {
                    "host": self.config_reader.get_property('redis.host', 'localhost'),
                    "port": int(self.config_reader.get_property('redis.port', '6379')),
                    "db": int(self.config_reader.get_property('redis.db', '0'))
                }
                
                    
                self.logger.info("Initializing PlatformListManager...")
                self.platform_list_manager = PlatformListManager(self.db, redis_config)
                self.logger.info("PlatformListManager initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize PlatformListManager: {str(e)}")
            self.platform_list_manager = None
        
        # Today's date for expiry check
        today = datetime.now().date()
        
        # Search for the ODFI in the platform lists
        results = self.platform_list_manager.search_entry("bank_routing_number", odfi_id)
        

        # First check if results contains any non-None values
        valid_results = [result for result in results if result is not None]

        # Check if found in any list
        if valid_results:
            for result in results:
                list_type = result['list_type']
                entity = result['entity']
                identifiers = result['identifiers']
                
                # Check for expired entries
                is_expired = False
                for identifier in identifiers:
                    if identifier['information_type'] == 'bank_routing_number' and identifier['information_value'] == odfi_id:
                        expiry_date = identifier.get('expiry_date')
                        if expiry_date:
                            # Convert to date object if it's a string
                            if isinstance(expiry_date, str):
                                try:
                                    expiry_date = datetime.strptime(expiry_date, "%Y-%m-%d").date()
                                except ValueError:
                                    continue
                            
                            # Skip if expired
                            if expiry_date < today:
                                is_expired = True
                                break
                
                # Skip expired entries
                if is_expired:
                    continue
                    
                # Check against blacklist
                if list_type == ListType.BLACKLIST or list_type == ListType.ODFI_SUSPICIOUS:
                    return blacklist_score
                
                # If it's in another list but has a risk score, use that
                if 'risk_score' in entity and entity['risk_score'] is not None:
                    base_score = max(base_score, float(entity['risk_score']))
        else:
            base_score= 0
        
        
        # Basic format checks (keep as a fallback)
        if not odfi_id.isdigit() or len(odfi_id) < 8:
            base_score = max(base_score, format_error_score)
        
        # Check for newly registered ODFIs
        # Search for entities with recent added_date
        thirty_days_ago = today - timedelta(days=30)
    
        try:
            self.db.reconnect()
            
            # Look for new entries in the platform list (entities added within last 30 days)
            new_odfi_query = """
            SELECT e.list_entity_id 
            FROM platform_list_items i
            JOIN platform_list_entities e ON i.list_entity_id = e.list_entity_id
            WHERE i.information_type = 'bank_routing_number'
            AND i.information_value = %s
            AND i.added_date >= %s
            AND (i.expiry_date IS NULL OR i.expiry_date >= CURRENT_DATE)
            LIMIT 1
            """
            results = self.db.execute_query(new_odfi_query, (odfi_id, thirty_days_ago.strftime("%Y-%m-%d")))

            # Check if any results were returned
            is_new_odfi = bool(results and len(results) > 0)

            if is_new_odfi:
                base_score = max(base_score, new_odfi_score)

            return base_score
        
        except Exception as e:
            self.logger.error(f"Error checking for new ODFI: {str(e)}")

        
        # Check for unusual volume pattern
        current_time = datetime.now()
        if self._is_off_hours(current_time):
            # Query for high volume ODFIs
            try:
                self.db.reconnect()

                volume_query = """
                SELECT e.list_entity_id 
                FROM platform_list_items i
                JOIN platform_list_entities e ON i.list_entity_id = e.list_entity_id
                WHERE i.information_type = 'bank_routing_number'
                AND i.information_value = %s
                AND e.frequency > %s
                AND (i.expiry_date IS NULL OR i.expiry_date >= CURRENT_DATE)
                LIMIT 1
                """
                
                high_volume_threshold = int(self.config_reader.get_property('odfi.high_volume_threshold', '100'))
                results = self.db.execute_query(volume_query, (odfi_id, high_volume_threshold))

                 # Check if any results were returned
                is_high_volume = bool(results and len(results) > 0)

                if is_high_volume:
                    base_score = max(base_score, high_volume_off_hours_score)

                return base_score
        
            except Exception as e:
                self.logger.error(f"Error checking for high volume ODFI: {str(e)}")

    
    def _is_off_hours(self, current_time: datetime) -> bool:
        """
        Check if current time is outside normal business hours
        
        :param current_time: Current datetime
        :return: Boolean indicating if outside business hours
        """
        # Use configuration values if available
        start_hour = self.ach_off_hours_start if hasattr(self, 'ach_off_hours_start') else 20
        end_hour = self.ach_off_hours_end if hasattr(self, 'ach_off_hours_end') else 6
        
        hour = current_time.hour
        
        # Check if hour is outside business hours
        # Example: if start=20, end=6, then hours 20, 21, 22, 23, 0, 1, 2, 3, 4, 5 are off-hours
        if start_hour > end_hour:  # Overnight case
            return hour >= start_hour or hour < end_hour
        else:  # Same day case
            return hour >= start_hour and hour < end_hour

    def detect_unusual_patterns(self, transaction: Dict, metrics: TransactionMetrics) -> List[Dict]:
        """Detect unusual patterns based on payment channel"""
        channel = transaction.get('channel', 'UNKNOWN')
        if channel not in self.channel_thresholds:
            return []
            
        alerts = []
        thresholds = self.channel_thresholds[channel]
        current_time = datetime.now()
        amount = float(transaction.get('amount', 0))
        country_code = transaction.get('debtor_country', 'UNKNOWN')
        

        # Channel-specific checks
        if channel == 'WIRE':
            alerts.extend(self._check_wire_patterns(transaction, metrics, current_time))
        elif channel == 'SWIFT':
            alerts.extend(self._check_swift_patterns(transaction, metrics, current_time))
        elif channel == 'FEDNOW':
            alerts.extend(self._check_fednow_patterns(transaction, metrics, current_time))
        elif channel == 'ZELLE':
            alerts.extend(self._check_zelle_patterns(transaction, metrics, current_time))
        elif channel == 'ACH':
            alerts.extend(self._check_ach_patterns(transaction,metrics,current_time))
            
        return alerts

    def _check_wire_patterns(self, transaction: Dict, metrics: TransactionMetrics, current_time: datetime) -> List[Dict]:
        """Check for unusual WIRE patterns"""
        alerts = []
        thresholds = self.channel_thresholds['WIRE']
        # Initialize local_time_desc to ensure it's always defined
        local_time_desc = ""

        # Get debtor country for time zone conversion
        country_code = transaction.get('debtor_country', 'UNKNOWN')
        
        # Check business hours
        if not self._is_business_hours(current_time, thresholds, country_code):
            local_time_desc = ""
            if country_code != 'UNKNOWN' and hasattr(self, 'timezone_manager'):
                local_time = self.timezone_manager.convert_to_local_time(current_time, country_code)
                local_time_desc = f" (local time in {country_code}: {local_time.strftime('%H:%M')})"
                
            alerts.append({
                'type': 'WIRE_OUTSIDE_BUSINESS_HOURS',
                'severity': 'MEDIUM',
                'details': f"Wire transfer initiated outside business hours ({thresholds['business_hours_start']}-{thresholds['business_hours_end']}){local_time_desc}"
            })
            
        # Check weekend
        is_weekend = False
        if country_code != 'UNKNOWN' and hasattr(self, 'timezone_manager'):
            local_time = self.timezone_manager.convert_to_local_time(current_time, country_code)
            is_weekend = local_time.weekday() >= 5
        else:
            is_weekend = current_time.weekday() >= 5


        if not thresholds['weekend_allowed'] and is_weekend:
            local_time_desc = ""
        
            if country_code != 'UNKNOWN' and hasattr(self, 'timezone_manager'):
                local_time = self.timezone_manager.convert_to_local_time(current_time, country_code)
                local_time_desc = f" (local time in {country_code}: {local_time.strftime('%A')})"
        
        alerts.append({
            'type': 'WIRE_WEEKEND_TRANSFER',
            'severity': 'HIGH',
            'details': f"Wire transfer initiated on weekend{local_time_desc}"
        })
            
        # Check high-value transfer
        if float(transaction.get('amount', 0)) > thresholds['high_amount']:
            alerts.append({
                'type': 'WIRE_HIGH_VALUE',
                'severity': 'HIGH',
                'details': f"High-value wire transfer exceeding {thresholds['high_amount']}"
            })
            
        return alerts

    def _check_swift_patterns(self, transaction: Dict, metrics: TransactionMetrics, current_time: datetime) -> List[Dict]:
        """Check for unusual SWIFT patterns"""
        alerts = []
        thresholds = self.channel_thresholds['SWIFT']
        # Initialize local_time_desc to ensure it's always defined
        local_time_desc = ""

        # Get debtor country for time zone conversion
        country_code = transaction.get('debtor_country', 'UNKNOWN')
        
        # Check business hours with time zone awareness
        if not self._is_business_hours(current_time, thresholds, country_code):
            local_time_desc = ""
            if country_code != 'UNKNOWN' and hasattr(self, 'timezone_manager'):
                local_time = self.timezone_manager.convert_to_local_time(current_time, country_code)
                local_time_desc = f" (local time in {country_code}: {local_time.strftime('%H:%M')})"
            alerts.append({
                'type': 'SWIFT_OUTSIDE_BUSINESS_HOURS',
                'severity': 'MEDIUM',
                'details': f"SWIFT transfer initiated outside business hours ({thresholds['business_hours_start']}-{thresholds['business_hours_end']}){local_time_desc}"
            })
            
            
        # Check weekend with time zone awareness
        is_weekend = False
        if country_code != 'UNKNOWN' and hasattr(self, 'timezone_manager'):
            local_time = self.timezone_manager.convert_to_local_time(current_time, country_code)
            is_weekend = local_time.weekday() >= 5
        else:
            is_weekend = current_time.weekday() >= 5
            
        if not thresholds['weekend_allowed'] and is_weekend:
            local_time_desc = ""
            if country_code != 'UNKNOWN' and hasattr(self, 'timezone_manager'):
                local_time = self.timezone_manager.convert_to_local_time(current_time, country_code)
                local_time_desc = f" (local time in {country_code}: {local_time.strftime('%A')})"
            alerts.append({
                'type': 'SWIFT_WEEKEND_TRANSFER',
                'severity': 'HIGH',
                'details': f"SWIFT transfer initiated on weekend{local_time_desc}"
            })
            
        # Check high-risk countries
        if metrics.high_risk_country:
            alerts.append({
                'type': 'SWIFT_HIGH_RISK_COUNTRY',
                'severity': 'HIGH',
                'details': "SWIFT transfer involving high-risk country"
            })
            
        return alerts
    
    def _check_ach_patterns(self, transaction: Dict, metrics: TransactionMetrics, current_time: datetime) -> List[Dict]:
        """Check for unusual ACH patterns"""
        alerts = []
        thresholds = self.channel_thresholds['ACH']
        # Initialize local_time_desc to ensure it's always defined
        local_time_desc = ""
        
        # Get debtor country for time zone conversion
        country_code = transaction.get('debtor_country', 'UNKNOWN')
        
        # Check business hours with time zone awareness
        if not self._is_business_hours(current_time, thresholds, country_code):
            local_time_desc = ""
            if country_code != 'UNKNOWN' and hasattr(self, 'timezone_manager'):
                local_time = self.timezone_manager.convert_to_local_time(current_time, country_code)
                local_time_desc = f" (local time in {country_code}: {local_time.strftime('%H:%M')})"
            alerts.append({
                'type': 'ACH_OUTSIDE_BUSINESS_HOURS',
                'severity': 'MEDIUM',
                'details': f"ACH transfer initiated outside business hours ({thresholds['business_hours_start']}-{thresholds['business_hours_end']}){local_time_desc}",
                'risk_score': float(self.config_reader.get_property('ach.risk.outside_hours', '40'))
            })
            
        # Check weekend with time zone awareness
        is_weekend = False
        if country_code != 'UNKNOWN' and hasattr(self, 'timezone_manager'):
            local_time = self.timezone_manager.convert_to_local_time(current_time, country_code)
            is_weekend = local_time.weekday() >= 5
        else:
            is_weekend = current_time.weekday() >= 5
            
        if not thresholds['weekend_allowed'] and is_weekend:
            local_time_desc = ""
            if country_code != 'UNKNOWN' and hasattr(self, 'timezone_manager'):
                local_time = self.timezone_manager.convert_to_local_time(current_time, country_code)
                local_time_desc = f" (local time in {country_code}: {local_time.strftime('%A')})"
            alerts.append({
                'type': 'ACH_WEEKEND_TRANSFER',
                'severity': 'MEDIUM',
                'details': f"ACH transfer initiated on weekend{local_time_desc}",
                'risk_score': float(self.config_reader.get_property('ach.risk.weekend', '40'))
            })
            
        # Check RDFI (Receiving Depository Financial Institution) pattern
        if transaction.get('receiving_dfi_id'):
            rdfi_risk_score = self._calculate_rdfi_risk(transaction['receiving_dfi_id'])
            if rdfi_risk_score > 0:
                # Determine severity based on risk score
                severity = 'LOW'
                if rdfi_risk_score >= 80:
                    severity = 'HIGH'
                elif rdfi_risk_score >= 50:
                    severity = 'MEDIUM'

                alerts.append({
                    'type': 'ACH_SUSPICIOUS_RDFI',
                    'severity': severity,
                    'details': f"ACH transfer to suspicious RDFI: {transaction['receiving_dfi_id']}",
                    'risk_score': rdfi_risk_score
                })
            
        # Check ODFI (Originating Depository Financial Institution) pattern
        if transaction.get('originator_dfi_id') and self._is_suspicious_odfi(transaction['originator_dfi_id']):
            alerts.append({
                'type': 'ACH_SUSPICIOUS_ODFI',
                'severity': 'HIGH',
                'details': f"ACH transfer from suspicious ODFI: {transaction['originator_dfi_id']}",
                'risk_score': float(self.config_reader.get_property('ach.risk.suspicious_odfi', '80'))
            })
            
        # Check for same-day ACH
        if transaction.get('same_day', False) and transaction.get('amount', 0) > thresholds.get('same_day_max_amount', 25000.0):
            alerts.append({
                'type': 'ACH_SAME_DAY_LARGE_AMOUNT',
                'severity': 'HIGH',
                'details': f"Same-day ACH transfer exceeds maximum amount: ${transaction.get('amount', 0):,.2f}",
                'risk_score': float(self.config_reader.get_property('ach.risk.same_day_large', '70'))
            })
            
        # Check for ACH return code
        if transaction.get('return_code') and transaction.get('return_code') in self.high_risk_return_codes:
            alerts.append({
                'type': 'ACH_HIGH_RISK_RETURN_CODE',
                'severity': 'HIGH',
                'details': f"ACH transfer with high-risk return code: {transaction.get('return_code')}",
                'risk_score': float(self.config_reader.get_property('ach.risk.high_risk_return', '90'))
            })
        
        return alerts

    def _check_fednow_patterns(self, transaction: Dict, metrics: TransactionMetrics, current_time: datetime) -> List[Dict]:
        """Check for unusual FedNow patterns"""
        alerts = []
        thresholds = self.channel_thresholds['FEDNOW']
        # Initialize local_time_desc to ensure it's always defined
        local_time_desc = ""
        
        # Get debtor country for time zone conversion
        country_code = transaction.get('debtor_country', 'UNKNOWN')
        
        # Check for rapid succession transactions
        if metrics.velocity_24h > thresholds['rapid_succession_count']:
            alerts.append({
                'type': 'FEDNOW_RAPID_SUCCESSION',
                'severity': 'HIGH',
                'details': f"Multiple FedNow transfers within {thresholds['rapid_transaction_interval']} minutes"
            })
            
        # Check for high-value instant payment
        if float(transaction.get('amount', 0)) > thresholds['high_amount']:
            alerts.append({
                'type': 'FEDNOW_HIGH_VALUE',
                'severity': 'HIGH',
                'details': f"High-value FedNow transfer exceeding {thresholds['high_amount']}"
            })
        
        # Add time zone aware business hours check for FedNow
        if hasattr(thresholds, 'business_hours_start') and hasattr(thresholds, 'business_hours_end'):
            if not self._is_business_hours(current_time, thresholds, country_code):
                local_time_desc = ""
                if country_code != 'UNKNOWN' and hasattr(self, 'timezone_manager'):
                    local_time = self.timezone_manager.convert_to_local_time(current_time, country_code)
                    local_time_desc = f" (local time in {country_code}: {local_time.strftime('%H:%M')})"
                alerts.append({
                    'type': 'FEDNOW_OUTSIDE_BUSINESS_HOURS',
                    'severity': 'MEDIUM',
                    'details': f"FedNow transfer initiated outside business hours ({thresholds['business_hours_start']}-{thresholds['business_hours_end']}){local_time_desc}"
                })
                
        return alerts

    def _check_zelle_patterns(self, transaction: Dict, metrics: TransactionMetrics, current_time: datetime) -> List[Dict]:
        """Check for unusual Zelle patterns"""
        alerts = []
        thresholds = self.channel_thresholds.get('ZELLE', {})
        # Initialize local_time_desc to ensure it's always defined
        local_time_desc = ""
    
        # Get debtor country for time zone conversion
        country_code = transaction.get('debtor_country', 'UNKNOWN')
        
        # Add time zone aware business hours check for Zelle
        if 'business_hours_start' in thresholds and 'business_hours_end' in thresholds:
            if not self._is_business_hours(current_time, thresholds, country_code):
                local_time_desc = ""
                if country_code != 'UNKNOWN' and hasattr(self, 'timezone_manager'):
                    local_time = self.timezone_manager.convert_to_local_time(current_time, country_code)
                    local_time_desc = f" (local time in {country_code}: {local_time.strftime('%H:%M')})"
                alerts.append({
                    'type': 'ZELLE_OUTSIDE_BUSINESS_HOURS',
                    'severity': 'MEDIUM',
                    'details': f"Zelle transfer initiated outside business hours ({thresholds['business_hours_start']}-{thresholds['business_hours_end']}){local_time_desc}"
                })
        
        # Check weekend with time zone awareness
        if 'weekend_allowed' in thresholds:
            is_weekend = False
            if country_code != 'UNKNOWN' and hasattr(self, 'timezone_manager'):
                local_time = self.timezone_manager.convert_to_local_time(current_time, country_code)
                is_weekend = local_time.weekday() >= 5
            else:
                is_weekend = current_time.weekday() >= 5
                
            if not thresholds['weekend_allowed'] and is_weekend:
                local_time_desc = ""
                if country_code != 'UNKNOWN' and hasattr(self, 'timezone_manager'):
                    local_time = self.timezone_manager.convert_to_local_time(current_time, country_code)
                    local_time_desc = f" (local time in {country_code}: {local_time.strftime('%A')})"
                alerts.append({
                    'type': 'ZELLE_WEEKEND_TRANSFER',
                    'severity': 'MEDIUM',
                    'details': f"Zelle transfer initiated on weekend{local_time_desc}"
                })

        # Check for multiple new recipients    
        max_new_recipients = float(self.config_reader.get_property('zelle.threshold.new_recipients_24h', default='5'))
        if metrics.unique_recipients_24h > max_new_recipients:
            alerts.append({
                'type': 'ZELLE_MULTIPLE_NEW_RECIPIENTS',
                'severity': self.config.get_property('zelle.threshold.new_recipients_severity', default='HIGH'),
                'details': f"Multiple new Zelle recipients within 24 hours: {metrics.unique_recipients_24h}"
            })

        # 2. Transaction Velocity Check (24h)
        max_velocity_24h = float(self.config_reader.get_property('zelle.threshold.recipient_total_24h', default='6'))
        if metrics.velocity_24h > max_velocity_24h:
            alerts.append({
                'type': 'ZELLE_TRANSACTION_VELOCITY_EXCEEDED',
                'severity': 'MEDIUM',
                'details': f"Transaction velocity in 24h exceeded: {metrics.velocity_24h}"
            })
            
        # 3. Daily Amount Check (24h)
        max_daily_amount = float(self.config_reader.get_property('zelle.threshold.max_daily_total_amount', default='5000'))
        if metrics.amount_24h > max_daily_amount:
            alerts.append({
                'type': 'ZELLE_DAILY_AMOUNT_EXCEEDED',
                'severity': 'HIGH',
                'details': f"Daily total amount exceeded: {metrics.amount_24h}"
            })

        # 4. Unusual Amount Check (using 30-day average and standard deviation)
        unusual_amount_multiplier = float(self.config_reader.get_property('zelle.threshold.unusual_high_amount_multiplier', default='2.5'))
        unusual_amount_threshold = metrics.avg_amount_30d + (unusual_amount_multiplier * metrics.std_amount_30d)
        
        
        current_transaction_amount = transaction.get('amount')  # This should be passed from the calling function
        if current_transaction_amount > unusual_amount_threshold:
            alerts.append({
                'type': 'ZELLE_UNUSUAL_HIGH_AMOUNT',
                'severity': 'HIGH',
                'details': f"Transaction amount unusually high: {current_transaction_amount} vs threshold {unusual_amount_threshold}"
            })

        # 5. New Recipient Check
        if metrics.new_recipient:
            alerts.append({
                'type': 'ZELLE_NEW_RECIPIENT',
                'severity': 'MEDIUM',
                'details': "Transaction involves a new recipient"
            })

        # 6. Cross-Border Transaction Check
        if metrics.cross_border:
            alerts.append({
                'type': 'ZELLE_CROSS_BORDER_TRANSACTION',
                'severity': 'HIGH',
                'details': "Cross-border transaction detected"
            })

        # 7. High-Risk Country Check
        if metrics.high_risk_country:
            alerts.append({
                'type': 'ZELLE_HIGH_RISK_COUNTRY',
                'severity': 'HIGH',
                'details': "Transaction involves a high-risk country"
            })

        return alerts
            
        

    def _is_business_hours(self, current_time: datetime, thresholds: Dict,
                           country_code: str = None) -> bool:
        
        """Check if current time is within business hours for given country"""
        if not hasattr(self, 'timezone_manager'):
            self.timezone_manager = TimeZoneManager()
            
        if country_code:
            return self.timezone_manager.is_business_hours(
                current_time, 
                country_code,
                thresholds['business_hours_start'],
                thresholds['business_hours_end']
            )
        else:
            # Fallback to old method if no country code
            hour = current_time.hour
            return thresholds['business_hours_start'] <= hour < thresholds['business_hours_end']
    


        # hour = current_time.hour
        # return thresholds['business_hours_start'] <= hour < thresholds['business_hours_end']


    def calculate_alert_risk_score(self, alerts: List[Dict]) -> Tuple[float, List[str]]:
        """
        Calculate risk score based on channel-specific alerts
        Reads configuration from config.properties
        
        :param alerts: List of alerts detected for a transaction
        :return: Tuple of (risk_score, contributing_factors)
        """
        try:
            # Ensure alerts is a list
            if not isinstance(alerts, list):
                alerts = []

            # Define severity weights 
            severity_weights = {
                'LOW': float(self.config_reader.get_property('channel.alert.severity.low', '0.2')),
                'MEDIUM': float(self.config_reader.get_property('channel.alert.severity.medium', '0.5')),
                'HIGH': float(self.config_reader.get_property('channel.alert.severity.high', '0.8')),
                'CRITICAL': float(self.config_reader.get_property('channel.alert.severity.critical', '1.0'))
            }
            
            # Read score calculation parameters
            score_multiplier = float(self.config_reader.get_property('channel.alert.score.multiplier', '20'))
            max_score = float(self.config_reader.get_property('channel.alert.max_score', '100'))

            # Track highest risk score from direct risk scores in alerts
            direct_risk_score = 0

            # Calculate weighted risk score for severity-based alerts
            severity_based_score = 0

            for alert in alerts:
                # Use direct risk score if available
                if 'risk_score' in alert and isinstance(alert['risk_score'], (int, float)):
                    direct_risk_score = max(direct_risk_score, float(alert['risk_score']))
                else:
                    # Fall back to severity-based calculation
                    severity = alert.get('severity', 'LOW').upper()
                    severity_based_score += severity_weights.get(severity, severity_weights['LOW'])
            
            # Scale severity-based score
            severity_based_score = severity_based_score * score_multiplier

            # Take the higher of direct risk score or severity-based score
            risk_score = max(direct_risk_score, severity_based_score)
            
            # Ensure score doesn't exceed maximum
            risk_score = min(risk_score, max_score)
            
            # Extract contributing factors (focusing on high and critical severity)
            contributing_factors = [
                f"{alert.get('type', 'UNKNOWN')}: {alert.get('details', 'No details')}"
                for alert in alerts
                # Include all alerts now for better visibility
            ]
            
            return risk_score, contributing_factors
        
        except Exception as e:
            logger.error(f"Error calculating alert risk score: {str(e)}")
            # Return default low-risk result
            return 0, []
    
class DecisionController:
    """Controller that makes decisions based on risk scores"""
    
    def __init__(self, config_reader: ConfigReader):
        self.config_reader = config_reader
        
        # Load thresholds from configuration
        self.thresholds = {
            'critical': float(self.config_reader.get_property('risk.threshold.critical_risk', '85')),
            'high': float(self.config_reader.get_property('risk.threshold.high_risk', '70')),
            'medium': float(self.config_reader.get_property('risk.threshold.medium_risk', '50')),
            'low': float(self.config_reader.get_property('risk.threshold.low_risk', '25')),
            'informational': float(self.config_reader.get_property('risk.threshold.informational', '10'))
        }
        
        # Load actions for each risk level
        self.actions = {
            'critical': self.config_reader.get_property('risk.action.critical', 'BLOCK_AND_REPORT'),
            'high': self.config_reader.get_property('risk.action.high', 'ALERT_AND_REVIEW'),
            'medium': self.config_reader.get_property('risk.action.medium', 'FLAG_FOR_REVIEW'),
            'low': self.config_reader.get_property('risk.action.low', 'MONITOR'),
            'informational': self.config_reader.get_property('risk.action.informational', 'LOG')
        }
    
    def get_risk_level(self, risk_score: float) -> str:
        """
        Determine risk level based on risk score
        
        :param risk_score: Calculated risk score (0-100)
        :return: Risk level as string
        """
        if risk_score >= self.thresholds['critical']:
            return 'critical'
        elif risk_score >= self.thresholds['high']:
            return 'high'
        elif risk_score >= self.thresholds['medium']:
            return 'medium'
        elif risk_score >= self.thresholds['low']:
            return 'low'
        elif risk_score >= self.thresholds['informational']:
            return 'informational'
        else:
            return 'normal'
    
    def get_action(self, risk_score: float) -> str:
        """
        Determine action based on risk score
        
        :param risk_score: Calculated risk score (0-100)
        :return: Action to take
        """
        risk_level = self.get_risk_level(risk_score)
        return self.actions.get(risk_level, 'PROCESS')
    
    def process_transaction_result(self, result: Dict) -> Dict:
        """
        Process transaction result and determine appropriate action
        
        :param result: Transaction analysis result
        :return: Updated result with decision information
        """
        risk_score = result.get('final_risk_score', 0)
        risk_level = self.get_risk_level(risk_score)
        action = self.get_action(risk_score)
        
        # Update result with decision information
        result.update({
            'risk_level': risk_level.upper(),
            'action': action,
            'review_required': action in ['BLOCK_AND_REPORT', 'ALERT_AND_REVIEW', 'FLAG_FOR_REVIEW'],
            'process_normally': action in ['MONITOR', 'LOG', 'PROCESS']
        })
        
        return result
# Module-level singleton for OFAC SDN list
_OFAC_SDN_LIST = None
_OFAC_LAST_MODIFIED = 0
_OFAC_NAME_INDEX = {}

def get_ofac_sdn_list(file_path='ofac_sdn.txt'):
    """Get or load the OFAC SDN list as a singleton"""
    global _OFAC_SDN_LIST, _OFAC_LAST_MODIFIED, _OFAC_NAME_INDEX
    
    # Check if file has been modified since last load
    try:
        current_mtime = os.path.getmtime(file_path)
    except:
        current_mtime = 0
        
    # If list is not loaded or file has changed, load it
    if _OFAC_SDN_LIST is None or current_mtime > _OFAC_LAST_MODIFIED:
        logger.info(f"Loading OFAC SDN list from {file_path}")
        sdn_entries = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Skip header if exists
                next(f, None)
                for line in f:
                    # Split by comma (same as your original function)
                    entry = line.strip().split(',')
                    if len(entry) >= 2:
                        sdn_entries.append({
                            'name': entry[0].strip().upper(),
                            'type': entry[1].strip().upper(),
                            'program': entry[2].strip().upper() if len(entry) > 2 else '',
                            'title': entry[3].strip().upper() if len(entry) > 3 else '',
                            'remarks': entry[4].strip().upper() if len(entry) > 4 else ''
                        })
            
            _OFAC_SDN_LIST = sdn_entries
            _OFAC_LAST_MODIFIED = current_mtime
            
            # Build the name index
            _OFAC_NAME_INDEX = build_name_index(sdn_entries)
            
            logger.info(f"Loaded {len(sdn_entries)} OFAC SDN entries")
        except FileNotFoundError:
            logger.error(f"OFAC SDN list file not found: {file_path}")
            _OFAC_SDN_LIST = []
        except Exception as e:
            logger.error(f"Error loading SDN list: {str(e)}")
            _OFAC_SDN_LIST = []
    
    return _OFAC_SDN_LIST, _OFAC_NAME_INDEX

def build_name_index(sdn_list):
    """Build an index for faster name lookups"""
    name_index = {}
    for i, entry in enumerate(sdn_list):
        name = entry.get('name', '')
        if name:
            # Index by first letter for faster filtering
            first_letter = name[0] if name else ''
            if first_letter not in name_index:
                name_index[first_letter] = []
            name_index[first_letter].append(i)
            
            # Also index by first letter of each word in multi-word names
            words = name.split()
            if len(words) > 1:
                for word in words[1:]:  # Skip first word as it's already indexed
                    if word:
                        first_letter = word[0]
                        if first_letter not in name_index:
                            name_index[first_letter] = []
                        if i not in name_index[first_letter]:
                            name_index[first_letter].append(i)
    
    return name_index
    
class OFACSDNChecker:
    def __init__(self, config_reader: ConfigReader, sdn_file_path: str = 'ofac_sdn.txt'):
        """
        Initialize OFAC SDN Checker
        
        :param config_reader: Configuration reader
        :param sdn_file_path: Path to the OFAC SDN list file
        """
        self.threshold_exact = 100
        self.threshold_high = 90
        self.threshold_medium = 80
        self.threshold_low = 70
        
        # Use singleton pattern for SDN list
        self.sdn_file_path = sdn_file_path
        self.sdn_list, self.name_index = get_ofac_sdn_list(sdn_file_path)
        
        self.logger = logging.getLogger(__name__)
        self.config_reader = config_reader
        
        # Create a name match cache (in-memory for simplicity)
        self.name_match_cache = {}
        self.max_cache_size = 1000  # Limit cache size to prevent memory issues

    # You don't need the _load_sdn_list method anymore

    def get_match_category(self, score):
        """Determine match category based on score"""
        if score >= self.threshold_exact:
            return 'EXACT_MATCH'
        elif score >= self.threshold_high:
            return 'HIGH_MATCH'
        elif score >= self.threshold_medium:
            return 'MEDIUM_MATCH'
        elif score >= self.threshold_low:
            return 'LOW_MATCH'
        else:
            return 'NO_MATCH'

    def _normalize_name(self, name) -> str:
        """
        Normalize a name for comparison
        
        :param name: Name to normalize
        :return: Normalized name string or empty string if normalization fails
        """
        try:
            # Handle None values
            if name is None:
                return ""
                
            # Convert to string if needed
            if not isinstance(name, str):
                try:
                    name = str(name)
                except:
                    return ""
            
            # Normalize whitespace
            name = re.sub(r'\s+', ' ', name).strip()
            
            # Convert to lowercase
            name = name.lower()
            
            # Remove non-alphanumeric characters except spaces
            name = re.sub(r'[^a-z0-9 ]', '', name)
            
            return name
        except Exception as e:
            self.logger.error(f"Error normalizing name: {str(e)}")
            return ""

    def check_name(self, name, threshold=None):
        """
        Check a name against the OFAC SDN list with enhanced matching
        
        Args:
            name (str): Name to check
            threshold (int, optional): Minimum match score to consider
            
        Returns:
            list: List of match dictionaries, sorted by match score
        """
        # Perform basic validation
        if not name or not isinstance(name, str):
            return []
            
        # Set default threshold if not provided
        if threshold is None:
            threshold = self.threshold_low  # Default to low threshold
        
        # Normalize the input name
        normalized_name = self._normalize_name(name)
        if not normalized_name:
            return []
        
        # Check cache first for exact match
        cache_key = f"{normalized_name}:{threshold}"
        if cache_key in self.name_match_cache:
            return self.name_match_cache[cache_key]
        
        matches = []
        start_time = time.time()
        
        # Use index for faster filtering - only check names that start with the same letter
        candidate_indices = []
        if normalized_name:
            # Get all words in the name
            words = normalized_name.split()
            for word in words:
                if word:
                    first_letter = word[0].upper()
                    if first_letter in self.name_index:
                        candidate_indices.extend(self.name_index[first_letter])
            
            # Remove duplicates
            candidate_indices = list(set(candidate_indices))
        
        # If no candidates found, use all entries (fallback)
        if not candidate_indices:
            candidate_indices = range(len(self.sdn_list))
        
        # Limit the number of candidates to check (if too many matches)
        if len(candidate_indices) > 1000:
            # If too many candidates, limit to first 1000 to prevent excessive processing
            candidate_indices = candidate_indices[:1000]
        
        # Check against filtered entries
        for idx in candidate_indices:
            sdn_entry = self.sdn_list[idx]
            sdn_name = sdn_entry.get('name', '')
            
            # Quick length filter - skip if length difference is too great
            if abs(len(normalized_name) - len(sdn_name)) > len(normalized_name) / 2:
                continue
                
            normalized_sdn_name = self._normalize_name(sdn_name)
            if not normalized_sdn_name:
                continue
            
            # Calculate match score using fuzzy matching
            import difflib
            match_ratio = difflib.SequenceMatcher(None, normalized_name, normalized_sdn_name).ratio() * 100
            
            # If score exceeds threshold, add to matches
            if match_ratio >= threshold:
                match_category = self.get_match_category(match_ratio)
                
                # Map match category to severity
                severity_map = {
                    'EXACT_MATCH': 'CRITICAL',
                    'HIGH_MATCH': 'HIGH',
                    'MEDIUM_MATCH': 'MEDIUM',
                    'LOW_MATCH': 'LOW',
                    'NO_MATCH': 'LOW'
                }
                
                # Analyze name components for additional match insights
                name_component_analysis = self._analyze_name_components(normalized_name, normalized_sdn_name)
                
                # Analyze business type
                # business_type_analysis = self._analyze_business_type(name, sdn_name)
                
                
                # Create match elements description
                match_elements = []
                match_reason = []
                
                # Add name match elements
                if match_ratio >= self.threshold_exact:
                    match_elements.append("Name (exact)")
                    match_reason.append("Exact name match")
                elif match_ratio >= self.threshold_high:
                    match_elements.append("Name (strong match)")
                    match_reason.append("Strong name similarity") 
                elif match_ratio >= self.threshold_medium:
                    match_elements.append("Name (partial)")
                    match_reason.append("Similar name")
                elif match_ratio >= self.threshold_low:
                    match_elements.append("Name (weak match)")
                    match_reason.append("Weak name similarity")
                    
                # Add component-specific matches
                if name_component_analysis['reversed_name_order']:
                    match_elements.append(f"Reversed name order (\"{name}\" vs \"{sdn_name}\")")
                    match_reason.append("Reversed name order")
                
                if name_component_analysis['surname_only_match']:
                    match_elements.append("Surname only")
                    match_reason.append("Surname match only")
                    
                # Add business type comparison
                # if business_type_analysis['different_business_type']:
                #     match_elements.append("Different business type")
                #     match_reason.append(f"Different business types ({business_type_analysis['query_type']} vs {business_type_analysis['sdn_type']})")
                
                # Create the match record with enhanced information
                matches.append({
                    'sdn_name': sdn_name,
                    'match_score': match_ratio,
                    'category': match_category,
                    'severity': severity_map.get(match_category, 'LOW'),
                    'entry_details': sdn_entry,
                    'match_elements': ", ".join(match_elements),
                    'match_reason': "; ".join(match_reason),
                    'name_component_analysis': name_component_analysis
                    # 'business_type_analysis': business_type_analysis
                })
        
        # Sort matches by score in descending order
        matches.sort(key=lambda x: x['match_score'], reverse=True)
        
        # Limit number of matches returned
        if len(matches) > 5:
            matches = matches[:5]
        
        # Cache the result
        if len(self.name_match_cache) < self.max_cache_size:
            self.name_match_cache[cache_key] = matches
        
        end_time = time.time()
        if end_time - start_time > 0.1:  # Log slow checks
            self.logger.warning(f"OFAC name check for '{name}' took {end_time - start_time:.3f}s")
        
        return matches

    def _analyze_name_components(self, normalized_name, normalized_sdn_name):
        """
        Analyze name components to detect patterns like reversed names, surname matches,
        and common business words.
        
        Args:
            normalized_name (str): Normalized query name
            normalized_sdn_name (str): Normalized SDN name
            
        Returns:
            dict: Analysis results including component matches
        """
        result = {
            'reversed_name_order': False,
            'surname_only_match': False,
            'common_business_words_only': False,
            'matching_components': []
        }
        
        # Split names into components
        query_parts = normalized_name.split()
        sdn_parts = normalized_sdn_name.split()
        
        # Skip analysis if either name has no parts
        if not query_parts or not sdn_parts:
            return result
        
        # Define common business words that could cause false positives
        common_business_words = [
            'international', 'inc', 'llc', 'ltd', 'limited', 'corporation', 'corp',
            'company', 'co', 'holdings', 'group', 'enterprises', 'trading', 'services',
            'consultants', 'solutions', 'industries', 'associates', 'partners', 'global'
        ]
        
        # Check for reversed name order (focusing on first and last parts)
        if len(query_parts) >= 2 and len(sdn_parts) >= 2:
            # Check if first and last parts are swapped
            if (query_parts[0] == sdn_parts[-1] and query_parts[-1] == sdn_parts[0]):
                result['reversed_name_order'] = True
                result['matching_components'].append('reversed_order')
            # For two-part names, check if they are reversed
            elif len(query_parts) == 2 and len(sdn_parts) == 2:
                if query_parts[0] == sdn_parts[1] and query_parts[1] == sdn_parts[0]:
                    result['reversed_name_order'] = True
                    result['matching_components'].append('reversed_order')
        
        # Check for surname-only match
        # Assume surname could be first part (Eastern convention) or last part (Western convention)
        if len(query_parts) >= 1 and len(sdn_parts) >= 1:
            # Check first part (Eastern convention)
            if query_parts[0] == sdn_parts[0] and len(query_parts) > 1 and len(sdn_parts) > 1:
                if query_parts[1:] != sdn_parts[1:]:  # Rest of the name is different
                    result['surname_only_match'] = True
                    result['matching_components'].append('first_part_match')
            
            # Check last part (Western convention)
            if query_parts[-1] == sdn_parts[-1] and len(query_parts) > 1 and len(sdn_parts) > 1:
                if query_parts[:-1] != sdn_parts[:-1]:  # Rest of the name is different
                    result['surname_only_match'] = True
                    result['matching_components'].append('last_part_match')
        
        # Check if only common business words are matching while key identifiers differ
        # First, identify all matching words
        matching_words = set(query_parts) & set(sdn_parts)
        
        # If there are matching words
        if matching_words:
            # Check if all matching words are common business words
            if all(word in common_business_words for word in matching_words):
                # Make sure there's at least one actual match
                if len(matching_words) > 0:
                    result['common_business_words_only'] = True
                    result['matching_components'].append('common_business_words')
                    result['matching_words'] = list(matching_words)
        
        # Handle specific case like "IAC INTERNATIONAL INC" vs "JIHAD INTERNATIONAL INC"
        if len(query_parts) >= 3 and len(sdn_parts) >= 3:
            # Extract the key identifier (typically the first word)
            query_identifier = query_parts[0]
            sdn_identifier = sdn_parts[0]
            
            # Check if identifiers are different but rest of the name matches
            if query_identifier != sdn_identifier:
                # Check if the rest of the words match
                rest_query = query_parts[1:]
                rest_sdn = sdn_parts[1:]
                
                if rest_query == rest_sdn and len(rest_query) > 0:
                    business_terms_only = all(word in common_business_words for word in rest_query)
                    
                    if business_terms_only:
                        result['common_business_words_only'] = True
                        result['matching_components'].append('different_core_name_same_business_terms')
                        result['different_identifiers'] = f"{query_identifier} vs {sdn_identifier}"
                        result['matching_terms'] = rest_query
        
        return result

    def _analyze_business_type(self, query_name, sdn_name):
        """
        Compare business types between query name and SDN name
        
        Args:
            query_name (str): Original query name
            sdn_name (str): SDN entry name
            
        Returns:
            dict: Analysis of business types
        """
        result = {
            'query_type': 'unknown',
            'sdn_type': 'unknown',
            'different_business_type': False
        }
        
        # Common business type indicators
        business_types = {
            'trading': ['trading', 'trade', 'import', 'export', 'international'],
            'consulting': ['consulting', 'consultants', 'advisors', 'services'],
            'technology': ['technology', 'tech', 'software', 'solutions', 'systems'],
            'financial': ['bank', 'financial', 'finance', 'investment', 'capital'],
            'manufacturing': ['manufacturing', 'industries', 'industrial', 'factory', 'production'],
            'shipping': ['shipping', 'logistics', 'freight', 'cargo', 'transport']
        }
        
        # Check for business type in query name
        query_lower = query_name.lower()
        for btype, indicators in business_types.items():
            for indicator in indicators:
                if indicator in query_lower:
                    result['query_type'] = btype
                    break
            if result['query_type'] != 'unknown':
                break
        
        # Check for business type in SDN name
        sdn_lower = sdn_name.lower()
        for btype, indicators in business_types.items():
            for indicator in indicators:
                if indicator in sdn_lower:
                    result['sdn_type'] = btype
                    break
            if result['sdn_type'] != 'unknown':
                break
        
        # Determine if business types are different
        if result['query_type'] != 'unknown' and result['sdn_type'] != 'unknown':
            result['different_business_type'] = (result['query_type'] != result['sdn_type'])
        
        return result

    def check_transaction_names(self, transaction: Dict, base_risk_score) -> Dict:
        """
        Check debtor and creditor names against OFAC SDN list
        
        :param transaction: Transaction dictionary
        :param base_risk_score: Base risk score for the transaction
        :return: Dictionary of OFAC check results
        """
        start_time = time.time()
        
        ofac_results = {
            'debtor_ofac_matches': [],
            'creditor_ofac_matches': [],
            'is_ofac_blocked': False,
            'match_score': 0,
            'severity': 'LOW'
        }
        
        try:
            # Safely get names with defaults
            debtor_name = str(transaction.get('debtor_name', '')).strip() if transaction.get('debtor_name') else ''
            creditor_name = str(transaction.get('creditor_name', '')).strip() if transaction.get('creditor_name') else ''
            
            if debtor_name:
                debtor_matches = self.check_name(debtor_name)
                if debtor_matches:
                    ofac_results['debtor_ofac_matches'] = debtor_matches
                    if debtor_matches[0].get('severity') in ['CRITICAL', 'HIGH']:
                        ofac_results['is_ofac_blocked'] = True
                        ofac_results['match_score'] = debtor_matches[0].get('match_score', 0)
                        ofac_results['severity'] = debtor_matches[0].get('severity', 'LOW')
            
            if creditor_name:
                creditor_matches = self.check_name(creditor_name)
                if creditor_matches:
                    ofac_results['creditor_ofac_matches'] = creditor_matches
                    if creditor_matches[0].get('severity') in ['CRITICAL', 'HIGH']:
                        ofac_results['is_ofac_blocked'] = True
                        ofac_results['match_score'] = max(
                            ofac_results['match_score'],
                            creditor_matches[0].get('match_score', 0)
                        )
                        ofac_results['severity'] = max(
                            ofac_results['severity'],
                            creditor_matches[0].get('severity', 'LOW'),
                            key=lambda x: {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2, 'CRITICAL': 3}.get(x, 0)
                        )
            
            # Log timing for OFAC check
            end_time = time.time()
            elapsed = end_time - start_time
            if elapsed > 0.2:  # Log if it took more than 200ms
                self.logger.warning(f"OFAC check took {elapsed:.3f}s for transaction")
            
        except Exception as e:
            self.logger.error(f"Error in check_transaction_names: {str(e)}")
        
        return ofac_results

    def _check_name_components(self, normalized_name, normalized_sdn_name):
        """Check for partial name matches including reversed order and surname-only matches"""
        
        # Split names into components
        name_parts = normalized_name.split()
        sdn_parts = normalized_sdn_name.split()
        
        results = {
            'has_surname_match': False,
            'has_reversed_order': False,
            'match_components': []
        }
        
        # Check for surname match (assuming first or last part could be surname in different cultures)
        if name_parts and sdn_parts:
            if name_parts[0] == sdn_parts[0] or name_parts[-1] == sdn_parts[-1]:
                results['has_surname_match'] = True
                results['match_components'].append('surname')
        
        # Check for reversed name order
        if len(name_parts) >= 2 and len(sdn_parts) >= 2:
            if (name_parts[0] == sdn_parts[-1] and name_parts[-1] == sdn_parts[0]) or \
            (name_parts[0] == sdn_parts[1] and name_parts[1] == sdn_parts[0]):
                results['has_reversed_order'] = True
                results['match_components'].append('reversed_order')
        
        return results
 
    def _extract_country_from_bank(self, bank_name):
        """
        Extract country from bank name or location string
        
        Args:
            bank_name (str): Bank name which might include location
            
        Returns:
            str: Extracted country name or empty string
        """
        import re
        
        if not bank_name or not isinstance(bank_name, str):
            return ""
        
        # Get country mappings from configuration
        country_mappings_str = self.config_reader.get_property('compliance_reporting.country_indicators', '')
        countries = {}
        
        # If configuration exists, parse it
        if country_mappings_str:
            try:
                import json
                countries = json.loads(country_mappings_str)
            except Exception as e:
                self.logger.error(f"Error parsing country indicators: {str(e)}")
                # Fall back to default mappings if needed
        
        # Default mappings if none in configuration
        if not countries:
            countries = {
                'UAE': ['uae', 'dubai', 'abu dhabi', 'sharjah', 'united arab emirates'],
                'Singapore': ['singapore', 'singaporean'],
                'China': ['china', 'chinese', 'shanghai', 'beijing', 'guangzhou', 'hong kong'],
                'Iran': ['iran', 'iranian', 'tehran'],
                'Russia': ['russia', 'russian', 'moscow', 'st petersburg'],
                'North Korea': ['north korea', 'dprk', 'pyongyang'],
                'Syria': ['syria', 'syrian', 'damascus'],
                'UK': ['uk', 'united kingdom', 'britain', 'british', 'london', 'england'],
                'USA': ['usa', 'united states', 'us', 'american', 'new york', 'chicago']
            }
        
        # Clean and standardize bank name
        bank_lower = bank_name.lower().strip()
        
        # Check for location in parentheses (common format)
        paren_match = re.search(r'\(([^)]+)\)', bank_lower)
        if paren_match:
            location = paren_match.group(1).strip()
            # Check if this location contains a country indicator
            for country, variations in countries.items():
                if any(variation in location for variation in variations):
                    return country
        
        # Check for location after comma (another common format)
        bank_parts = bank_lower.split(',')
        if len(bank_parts) > 1:
            location_part = bank_parts[1].strip()
            for country, variations in countries.items():
                if any(variation in location_part for variation in variations):
                    return country
        
        # Check for country name at end of bank name (e.g. "Bank of China")
        for country, variations in countries.items():
            for variation in variations:
                # Use word boundary to avoid partial matches
                pattern = r'\b' + re.escape(variation) + r'\b'
                if re.search(pattern, bank_lower):
                    return country
        
        # As a fallback, check for country indicators anywhere in the name
        for country, variations in countries.items():
            for variation in variations:
                if variation in bank_lower:
                    return country
        
        return ""

    def _evaluate_jurisdiction_risk(self, country, sanctions_program=None):
        """
        Evaluate the risk level of a jurisdiction using configuration settings
        
        Args:
            country (str): Country name to evaluate
            sanctions_program (str, optional): Specific sanctions program to check against
            
        Returns:
            dict: Risk assessment of the jurisdiction
        """
        result = {
            'country': country,
            'is_high_risk': False,
            'is_medium_risk': False,
            'is_low_risk': False,
            'risk_description': ''
        }
        
        if not country:
            result['risk_description'] = "Unknown location - cannot assess jurisdiction risk"
            return result
        
        # Get lists from configuration
        high_risk_countries_str = self.config_reader.get_property("compliance_reporting.jurisdictions.high_risk", '')
        medium_risk_countries_str = self.config_reader.get_property("compliance_reporting.jurisdictions.medium_risk", '')
        low_risk_countries_str = self.config_reader.get_property("compliance_reporting.jurisdictions.low_risk", '')
        
        # Parse comma-separated lists
        high_risk_countries = [c.strip() for c in high_risk_countries_str.split(',')] if high_risk_countries_str else []
        medium_risk_countries = [c.strip() for c in medium_risk_countries_str.split(',')] if medium_risk_countries_str else []
        low_risk_countries = [c.strip() for c in low_risk_countries_str.split(',')] if low_risk_countries_str else []
        
        # Check high-risk countries
        if country in high_risk_countries:
            result['is_high_risk'] = True
            program_text = f" for {sanctions_program}" if sanctions_program else ""
            result['risk_description'] = f"{country} is a high-risk jurisdiction{program_text}"
            return result
        
        # Check medium-risk countries
        if country in medium_risk_countries:
            result['is_medium_risk'] = True
            result['risk_description'] = f"{country} is a medium-risk jurisdiction for sanctions"
            return result
        
        # Check low-risk countries
        if country in low_risk_countries:
            result['is_low_risk'] = True
            program_text = f" for {sanctions_program}" if sanctions_program else ""
            result['risk_description'] = f"{country} is generally considered a low-risk jurisdiction{program_text}"
            return result
        
        # Default case - unknown risk
        result['risk_description'] = f"Risk level for {country} is not specifically categorized"
        return result

    def _evaluate_transaction_amount(self, amount, typical_threshold=None):
        """
        Evaluate transaction amount against thresholds from config
        
        Args:
            amount (float): Transaction amount
            typical_threshold (float, optional): Custom threshold for comparison
            
        Returns:
            dict: Assessment of the transaction amount
        """
        result = {
            'amount': amount,
            'exceeds_threshold': False,
            'threshold_description': ''
        }
        
        # Get thresholds from config
        threshold_low = float(self.config_reader.get_property('compliance_reporting.threshold.low', '10000'))
        threshold_medium = float(self.config_reader.get_property('compliance_reporting.threshold.medium', '50000'))
        threshold_high = float(self.config_reader.get_property('compliance_reporting.threshold.high', '100000'))
        
        # Use provided threshold or default medium
        threshold = typical_threshold if typical_threshold else threshold_medium
        
        # Evaluate against threshold
        if amount >= threshold_high:
            result['exceeds_threshold'] = True
            result['threshold_description'] = f"Transaction amount (${amount:,.2f}) exceeds high-value threshold (${threshold_high:,.2f})"
        elif amount >= threshold_medium:
            result['exceeds_threshold'] = True
            result['threshold_description'] = f"Transaction amount (${amount:,.2f}) exceeds medium-value threshold (${threshold_medium:,.2f})"
        elif amount >= threshold_low:
            result['exceeds_threshold'] = True
            result['threshold_description'] = f"Transaction amount (${amount:,.2f}) exceeds standard reporting threshold (${threshold_low:,.2f})"
        else:
            result['threshold_description'] = f"Transaction amount (${amount:,.2f}) is below suspicious threshold"
        
        return result

    def format_ofac_match_output(self, match_data, transaction):
        """
        Format OFAC match data into a standardized output
        
        Args:
            match_data (dict): Match data from check_name method
            transaction (dict): Transaction details
            
        Returns:
            str: Formatted OFAC match output
        """
        import datetime
        import random
        
        # Generate match ID with risk level indicator
        category = match_data.get('category', 'NO_MATCH')
        risk_indicator = category[0] if category else 'N'
        
        
        # Map category to risk level for display
        risk_level_map = {
            'EXACT_MATCH': 'Critical Risk',
            'HIGH_MATCH': 'High Risk',
            'MEDIUM_MATCH': 'Medium Risk',
            'LOW_MATCH': 'Low Risk',
            'NO_MATCH': 'No Risk'
        }
        
        # Map threshold ranges based on risk level
        threshold_ranges = {
            'Critical Risk': '100%',
            'High Risk': '90-99%',
            'Medium Risk': '80-89%',
            'Low Risk': '70-79%',
            'No Risk': '<70%'
        }
        
        # Generate match ID with risk level indicator
        risk_indicator = match_data.get('category', 'NO_MATCH')[0] if match_data.get('category') else 'N'
        match_id = f"OFAC-{risk_indicator}-{datetime.datetime.now().strftime('%Y-%m-%d')}"
        
        # Format current date
        match_date = datetime.datetime.now().strftime("%B %d, %Y")
        
        # Get match score and risk level
        match_score = match_data.get('match_score', 0)
        risk_level = risk_level_map.get(match_data.get('category', 'NO_MATCH'), 'No Risk')
        threshold_range = threshold_ranges.get(risk_level, '<70%')
        
        # Extract OFAC entity details
        sdn_entry = match_data.get('entry_details', {})
        sdn_name = match_data.get('sdn_name', 'N/A')
        
        # Extract program information - might need to be modified based on your actual SDN data structure
        sdn_type = sdn_entry.get('entity_type', 'Entity')
        program = sdn_entry.get('program', 'Unknown Sanctions Program')
        
        # Determine match reason and elements based on score
        match_reason = "No match"
        match_elements = "None"
        
        if match_score >= self.threshold_exact:
            match_reason = "Exact name match"
            match_elements = "Name (exact), Location (exact)"
        elif match_score >= self.threshold_high:
            match_reason = "Strong name similarity"
            match_elements = "Name (strong), Location (possible)"
        elif match_score >= self.threshold_medium:
            match_reason = "Similar name, similar location"
            match_elements = "Name (partial), Location (country match)"
        elif match_score >= self.threshold_low:
            match_reason = "Partial name similarity"
            match_elements = "Name (partial)"
        
        # Format the output
        output = f"""Match ID: {match_id}\n Match Status: {risk_level} \n Match Date: {match_date}\n Match Score: {match_score:.1f}% ({threshold_range}) \n\n Matched Entity Details:\n Name on OFAC List: {sdn_name}\nSDN Type: {sdn_type} \nList Program: {program} \nMatch Reason: {match_reason} \nMatch Elements: {match_elements} \n\nTransaction Details:\nTransaction ID: {transaction.get('transaction_id', 'Unknown')}\nTransaction Date: {transaction.get('transaction_date', 'Unknown')}\nAmount: ${transaction.get('amount', 0):,.2f} USD\nTransaction Type: {transaction.get('transaction_type', 'Unknown')}\nOriginator: {transaction.get('debtor_name', 'Unknown')}\nOriginator Account: {transaction.get('debtor_account', 'Unknown')}\nBeneficiary: {transaction.get('creditor_name', 'Unknown')}\nBeneficiary Account: {transaction.get('creditor_account', 'Unknown')}\nBeneficiary Bank: {transaction.get('creditor_bank', 'Unknown')}\nPayment Purpose: '{transaction.get('payment_purpose', 'Unknown')}'\n\nRisk Factors:\n
        """
        
        # Determine risk factors based on risk level
        risk_factors = []
        beneficiary_name = transaction.get('creditor_name', '')
        originator_name = transaction.get('debtor_name', '')
        # Logic to determine which party matched with the SDN entity
        if match_data.get('party_type') == 'creditor':
            matched_party = beneficiary_name
        elif match_data.get('party_type') == 'debtor':
            matched_party = originator_name
        else:
            # If not explicitly specified, determine based on higher match score
            creditor_score = 0
            debtor_score = 0
            
            # Check if we have scores for both parties
            if 'creditor_match_score' in match_data:
                creditor_score = match_data.get('creditor_match_score', 0)
            if 'debtor_match_score' in match_data:
                debtor_score = match_data.get('debtor_match_score', 0)
                
            # If we don't have specific scores, use fuzzy matching to determine
            if creditor_score == 0 and debtor_score == 0:
                import difflib
                creditor_score = difflib.SequenceMatcher(None, self._normalize_name(beneficiary_name), 
                                                        self._normalize_name(match_data.get('sdn_name', ''))).ratio() * 100
                debtor_score = difflib.SequenceMatcher(None, self._normalize_name(originator_name),
                                                    self._normalize_name(match_data.get('sdn_name', ''))).ratio() * 100
            
            # Choose the party with the higher match score
            matched_party = beneficiary_name if creditor_score >= debtor_score else originator_name
        

        risk_factors = []

        # Add name similarity factor based on match level
        if match_score >= self.threshold_exact:
            risk_factors.append(f"Exact name match between '{matched_party}' and sanctioned entity '{sdn_name}'")
        elif match_score >= self.threshold_high:
            risk_factors.append(f"Strong name similarity between '{matched_party}' and sanctioned entity '{sdn_name}'")
        elif match_score >= self.threshold_medium:
            risk_factors.append(f"Name similarity between '{matched_party}' and sanctioned entity '{sdn_name}'")
        else:
            risk_factors.append(f"Partial name similarity between '{matched_party}' and sanctioned entity '{sdn_name}'")

        # Add component-specific match details if available
        name_component_analysis = match_data.get('name_component_analysis', {})
        if name_component_analysis:
            if name_component_analysis.get('reversed_name_order'):
                risk_factors.append(f"Reversed name order detected (\"{matched_party}\" vs \"{sdn_name}\")")
            if name_component_analysis.get('surname_only_match'):
                risk_factors.append(f"Only surname matches between entities")
            if name_component_analysis.get('common_business_words_only'):
                matching_terms = ', '.join(name_component_analysis.get('matching_terms', ['common terms']))
                risk_factors.append(f"Match based only on common business terms: {matching_terms}")

        # Business type comparison
        # business_type_analysis = match_data.get('business_type_analysis', {})
        # if business_type_analysis:
        #     query_type = business_type_analysis.get('query_type', 'unknown')
        #     sdn_type = business_type_analysis.get('sdn_type', 'unknown')
            
        #     if query_type != 'unknown' and sdn_type != 'unknown':
        #         if query_type == sdn_type:
        #             if match_score >= self.threshold_medium:
        #                 risk_factors.append(f"Same industry as sanctioned entity ({query_type})")
        #             else:
        #                 risk_factors.append(f"Similar industry type ({query_type})")
        #         else:
        #             risk_factors.append(f"Different business activity ({query_type}) than sanctioned entity ({sdn_type})")

        # Jurisdiction analysis
        jurisdiction = self._extract_country_from_bank(transaction.get('creditor_bank', ''))
        jurisdiction_analysis = self._evaluate_jurisdiction_risk(jurisdiction, sdn_entry.get('program'))

        if jurisdiction_analysis:
            if jurisdiction_analysis.get('is_high_risk'):
                if match_score >= self.threshold_high:
                    risk_factors.append(f"Direct connection to sanctioned country ({jurisdiction})")
                else:
                    risk_factors.append(jurisdiction_analysis.get('risk_description', 
                                        f"Transaction involves high-risk jurisdiction ({jurisdiction})"))
            elif jurisdiction_analysis.get('is_medium_risk'):
                risk_factors.append(f"Geographic proximity to sanctioned region ({jurisdiction})")
            elif jurisdiction_analysis.get('is_low_risk'):
                risk_factors.append(jurisdiction_analysis.get('risk_description',
                                    f"Low-risk jurisdiction ({jurisdiction})"))
            else:
                risk_factors.append(f"Jurisdiction ({jurisdiction}) not specifically categorized for risk")

        # Amount analysis
        amount = float(transaction.get('amount', 0))
        amount_analysis = self._evaluate_transaction_amount(amount)

        if amount_analysis:
            if match_score >= self.threshold_exact:
                risk_factors.append(f"Transaction amount (${amount:,.2f}) consistent with known sanctioned activity")
            elif match_score >= self.threshold_high:
                risk_factors.append(f"Transaction amount (${amount:,.2f}) consistent with potential sanctioned activity")
            elif amount_analysis.get('exceeds_threshold'):
                risk_factors.append(amount_analysis.get('threshold_description', 
                                    f"Transaction size (${amount:,.2f}) exceeds typical threshold"))
            else:
                risk_factors.append(amount_analysis.get('threshold_description',
                                    f"Transaction amount (${amount:,.2f}) below suspicious threshold"))

        # Add sanctions program specific factor for exact and high matches
        if match_score >= self.threshold_high:
            program = sdn_entry.get('program', 'Unknown Sanctions Program')
            if match_score >= self.threshold_exact:
                risk_factors.append(f"Transaction clearly violates {program}")
            else:
                risk_factors.append(f"Transaction potentially implicates {program}")
        
        # Add risk factors to output
        for factor in risk_factors:
            output += f"- {factor}\n"
        
        if match_score >= self.threshold_exact:
            action_text = self.config_reader.get_property('compliance_reporting.actions.exact', 
                        "Suggest to Block Transaction immediately. Regulatory filing to be initiated. Please escalate to Legal, Compliance, and OFAC reporting teams. Suggest account to be placed on hold pending investigation.")
            output += action_text
        elif match_score >= self.threshold_high:
            action_text = self.config_reader.get_property('compliance_reporting.actions.high',
                        "Suggest to Block Transaction. Escalate to Sanctions Compliance Team for investigation. Submit notification to OFAC. Request additional information from customer.")
            output += action_text
        elif match_score >= self.threshold_medium:
            action_text = self.config_reader.get_property('compliance_reporting.actions.medium',
                        "Place transaction on 24-hour hold, pending enhanced due diligence review and determination by OFAC compliance officer.")
            output += action_text
        elif match_score >= self.threshold_low:
            action_text = self.config_reader.get_property('compliance_reporting.actions.low',
                        "Suggest to approve transaction to proceed with normal monitoring. Add note to customer file for future reference.")
            output += action_text
        else:
            action_text = self.config_reader.get_property('compliance_reporting.actions.none',
                        "No action required. Transaction to be processed normally.")
            output += action_text
        
        return output.strip()

    # Add this method to generate complete OFAC check output for a transaction
    def generate_ofac_report(self, transaction):
        """
        Generate a complete OFAC check report for a transaction
        
        Args:
            transaction (dict): Transaction details
        
        Returns:
            dict: Dictionary with formatted report and match data
        """
        # Perform OFAC checks
        ofac_results = self.check_transaction_names(transaction, 0)
        
        # Determine which matches to format (debtor or creditor)
        formatted_output = None
        match_data = None
        
        # First check debtor matches
        if ofac_results.get('debtor_ofac_matches'):
            debtor_matches = ofac_results.get('debtor_ofac_matches', [])
            if debtor_matches:
                match_data = debtor_matches[0]
                formatted_output = self.format_ofac_match_output(match_data, transaction)
        
        # If no significant debtor matches, check creditor matches
        if not formatted_output and ofac_results.get('creditor_ofac_matches'):
            creditor_matches = ofac_results.get('creditor_ofac_matches', [])
            if creditor_matches:
                match_data = creditor_matches[0]
                formatted_output = self.format_ofac_match_output(match_data, transaction)
        
        # If no matches at all, create a "no match" report
        if not formatted_output or len(formatted_output.strip()) < 50 or "Match ID:" not in formatted_output:
            # Create empty match data for no match case
            empty_match = {
                'match_score': 0,
                'category': 'NO_MATCH',
                'severity': 'LOW',
                'sdn_name': 'None',
                'entry_details': {}
            }
            formatted_output = self.format_ofac_match_output(empty_match, transaction)
        
        return {
            'formatted_report': formatted_output,
            'match_data': match_data,
            'raw_results': ofac_results
        }

    
class ISO20022MessageValidator:
    def __init__(self):
        self.namespaces = {
            'pain': 'urn:iso:std:iso:20022:tech:xsd:pain.001.001.09',
            'pacs': 'urn:iso:std:iso:20022:tech:xsd:pacs.008.001.08',
            'camt': 'urn:iso:std:iso:20022:tech:xsd:camt.054.001.08'
        }
        
    def detect_message_type(self, root: ET.Element) -> ISO20022MessageType:
        try:
            # Get the root tag without namespace
            tag = root.tag.split('}')[-1]
            if 'Document' not in tag:
                raise ValueError(f"Invalid root element: {tag}")
            
            # Get the namespace
            ns = root.tag.split('}')[0].strip('{')
            
            if 'pain.001' in ns:
                return ISO20022MessageType.PAIN_001
            elif 'pacs.008' in ns:
                return ISO20022MessageType.PACS_008
            elif 'camt.054' in ns:
                return ISO20022MessageType.CAMT_054
            
            raise ValueError(f"Unsupported message type: {ns}")
            
        except Exception as e:
            logging.error(f"Error detecting message type: {str(e)}")
            raise
    
    def validate_structure(self, root: ET.Element, message_type: ISO20022MessageType) -> bool:
        """Validates the presence of required elements in the given XML root."""
        """Validates the presence of required elements with improved error handling"""
        try:
            required_elements = self._get_required_elements(message_type)
            
            # Determine namespace prefix
            ns_prefix = {
                ISO20022MessageType.PAIN_001: 'pain',
                ISO20022MessageType.PACS_008: 'pacs',
                ISO20022MessageType.CAMT_054: 'camt'
            }.get(message_type)
            
            if not ns_prefix:
                raise ValueError(f"Unsupported message type: {message_type}")

            ns_uri = self.namespaces.get(ns_prefix)
            if not ns_uri:
                raise ValueError(f"Namespace not found for prefix: {ns_prefix}")

            ns = {ns_prefix: ns_uri}

            for xpath in required_elements:
                try:
                    # Split and reconstruct xpath with namespace
                    parts = xpath.split('/')
                    ns_xpath = './/' + '/'.join(f'{ns_prefix}:{part}' for part in parts)
                    
                    element = root.find(ns_xpath, namespaces=ns)
                    if element is None:
                        logger.warning(f"Missing required element: {xpath}")
                        return False
                except Exception as e:
                    logger.error(f"Error checking element {xpath}: {str(e)}")
                    return False

            return True
            
        except Exception as e:
            logger.error(f"Error validating structure: {str(e)}")
            return False
        
    def _get_required_elements(self, message_type: ISO20022MessageType) -> List[str]:
        """Get required elements based on message type"""
        if message_type == ISO20022MessageType.PAIN_001:
            return [
                'GrpHdr/MsgId',
                'GrpHdr/CreDtTm',
                'GrpHdr/NbOfTxs',
                'PmtInf/PmtInfId',
                'PmtInf/PmtMtd'
            ]
        elif message_type == ISO20022MessageType.PACS_008:
            return [
                'GrpHdr/MsgId',
                'GrpHdr/CreDtTm',
                'GrpHdr/NbOfTxs'
            ]
        elif message_type == ISO20022MessageType.CAMT_054:
            return [
                'BkToCstmrDbtCdtNtfctn/GrpHdr/MsgId',
                'BkToCstmrDbtCdtNtfctn/GrpHdr/CreDtTm'
            ]
        return []

class TimeZoneManager:
    """Manages time zone conversions based on country codes"""
    
    def __init__(self):
        # Map of ISO country codes to primary time zones
        self.country_to_timezone = { 
            'AD': 'Europe/Andorra',
            'AE': 'Asia/Dubai',
            'AF': 'Asia/Kabul',
            'AG': 'America/Antigua',
            'AI': 'America/Anguilla',
            'AL': 'Europe/Tirane',
            'AM': 'Asia/Yerevan',
            'AO': 'Africa/Luanda',
            'AQ': 'Antarctica/Casey',
            'AQ': 'Antarctica/Davis',
            'AQ': 'Antarctica/DumontDUrville',
            'AQ': 'Antarctica/Macquarie',
            'AQ': 'Antarctica/Mawson',
            'AQ': 'Antarctica/Palmer',
            'AQ': 'Antarctica/Rothera',
            'AQ': 'Antarctica/Syowa',
            'AQ': 'Antarctica/Troll',
            'AQ': 'Antarctica/Vostok',
            'AR': 'America/Argentina/Buenos_Aires',
            'AR': 'America/Argentina/Catamarca',
            'AR': 'America/Argentina/Cordoba',
            'AR': 'America/Argentina/Jujuy',
            'AR': 'America/Argentina/La_Rioja',
            'AR': 'America/Argentina/Mendoza',
            'AR': 'America/Argentina/Rio_Gallegos',
            'AR': 'America/Argentina/Salta',
            'AR': 'America/Argentina/San_Juan',
            'AR': 'America/Argentina/San_Luis',
            'AR': 'America/Argentina/Tucuman',
            'AR': 'America/Argentina/Ushuaia',
            'AS': 'Pacific/Pago_Pago',
            'AT': 'Europe/Vienna',
            'AU': 'Antarctica/Macquarie',
            'AU': 'Australia/Adelaide',
            'AU': 'Australia/Brisbane',
            'AU': 'Australia/Broken_Hill',
            'AU': 'Australia/Darwin',
            'AU': 'Australia/Eucla',
            'AU': 'Australia/Hobart',
            'AU': 'Australia/Lindeman',
            'AU': 'Australia/Lord_Howe',
            'AU': 'Australia/Melbourne',
            'AU': 'Australia/Perth',
            'AU': 'Australia/Sydney',
            'AW': 'America/Aruba',
            'AX': 'Europe/Mariehamn',
            'AZ': 'Asia/Baku',
            'BA': 'Europe/Sarajevo',
            'BB': 'America/Barbados',
            'BD': 'Asia/Dhaka',
            'BE': 'Europe/Brussels',
            'BF': 'Africa/Ouagadougou',
            'BG': 'Europe/Sofia',
            'BH': 'Asia/Bahrain',
            'BI': 'Africa/Bujumbura',
            'BJ': 'Africa/Porto-Novo',
            'BL': 'America/St_Barthelemy',
            'BM': 'Atlantic/Bermuda',
            'BN': 'Asia/Brunei',
            'BO': 'America/La_Paz',
            'BQ': 'America/Kralendijk',
            'BR': 'America/Araguaina',
            'BR': 'America/Bahia',
            'BR': 'America/Belem',
            'BR': 'America/Boa_Vista',
            'BR': 'America/Campo_Grande',
            'BR': 'America/Cuiaba',
            'BR': 'America/Eirunepe',
            'BR': 'America/Fortaleza',
            'BR': 'America/Maceio',
            'BR': 'America/Manaus',
            'BR': 'America/Noronha',
            'BR': 'America/Porto_Velho',
            'BR': 'America/Recife',
            'BR': 'America/Rio_Branco',
            'BR': 'America/Santarem',
            'BR': 'America/Sao_Paulo',
            'BS': 'America/Nassau',
            'BT': 'Asia/Thimphu',
            'BW': 'Africa/Gaborone',
            'BY': 'Europe/Minsk',
            'BZ': 'America/Belize',
            'CA': 'America/Atikokan',
            'CA': 'America/Blanc-Sablon',
            'CA': 'America/Cambridge_Bay',
            'CA': 'America/Creston',
            'CA': 'America/Dawson',
            'CA': 'America/Dawson_Creek',
            'CA': 'America/Edmonton',
            'CA': 'America/Fort_Nelson',
            'CA': 'America/Glace_Bay',
            'CA': 'America/Goose_Bay',
            'CA': 'America/Halifax',
            'CA': 'America/Inuvik',
            'CA': 'America/Iqaluit',
            'CA': 'America/Moncton',
            'CA': 'America/Nipigon',
            'CA': 'America/Pangnirtung',
            'CA': 'America/Rainy_River',
            'CA': 'America/Rankin_Inlet',
            'CA': 'America/Regina',
            'CA': 'America/Resolute',
            'CA': 'America/St_Johns',
            'CA': 'America/Swift_Current',
            'CA': 'America/Thunder_Bay',
            'CA': 'America/Toronto',
            'CA': 'America/Vancouver',
            'CA': 'America/Whitehorse',
            'CA': 'America/Winnipeg',
            'CA': 'America/Yellowknife',
            'CC': 'Indian/Cocos',
            'CD': 'Africa/Kinshasa',
            'CD': 'Africa/Lubumbashi',
            'CF': 'Africa/Bangui',
            'CG': 'Africa/Brazzaville',
            'CH': 'Europe/Zurich',
            'CI': 'Africa/Abidjan',
            'CK': 'Pacific/Rarotonga',
            'CL': 'America/Punta_Arenas',
            'CL': 'America/Santiago',
            'CL': 'Pacific/Easter',
            'CM': 'Africa/Douala',
            'CN': 'Asia/Shanghai',
            'CO': 'America/Bogota',
            'CR': 'America/Costa_Rica',
            'CU': 'America/Havana',
            'CV': 'Atlantic/Cape_Verde',
            'CW': 'America/Curacao',
            'CX': 'Indian/Christmas',
            'CY': 'Asia/Famagusta',
            'CY': 'Asia/Nicosia',
            'CZ': 'Europe/Prague',
            'DE': 'Europe/Berlin',
            'DE': 'Europe/Busingen',
            'DJ': 'Africa/Djibouti',
            'DK': 'Europe/Copenhagen',
            'DM': 'America/Dominica',
            'DO': 'America/Santo_Domingo',
            'DZ': 'Africa/Algiers',
            'EC': 'America/Guayaquil',
            'EC': 'Pacific/Galapagos',
            'EE': 'Europe/Tallinn',
            'EG': 'Africa/Cairo',
            'EH': 'Africa/El_Aaiun',
            'ER': 'Africa/Asmara',
            'ES': 'Africa/Ceuta',
            'ES': 'Atlantic/Canary',
            'ES': 'Europe/Madrid',
            'ET': 'Africa/Addis_Ababa',
            'FI': 'Europe/Helsinki',
            'FJ': 'Pacific/Fiji',
            'FK': 'Atlantic/Stanley',
            'FM': 'Pacific/Chuuk',
            'FM': 'Pacific/Kosrae',
            'FM': 'Pacific/Pohnpei',
            'FO': 'Atlantic/Faroe',
            'FR': 'Europe/Paris',
            'GA': 'Africa/Libreville',
            'GB': 'Europe/Guernsey',
            'GB': 'Europe/Isle_of_Man',
            'GB': 'Europe/Jersey',
            'GB': 'Europe/London',
            'GD': 'America/Grenada',
            'GE': 'Asia/Tbilisi',
            'GF': 'America/Cayenne',
            'GG': 'Europe/Guernsey',
            'GH': 'Africa/Accra',
            'GI': 'Europe/Gibraltar',
            'GL': 'America/Danmarkshavn',
            'GL': 'America/Godthab',
            'GL': 'America/Scoresbysund',
            'GL': 'America/Thule',
            'GM': 'Africa/Banjul',
            'GN': 'Africa/Conakry',
            'GP': 'America/Guadeloupe',
            'GQ': 'Africa/Malabo',
            'GR': 'Europe/Athens',
            'GS': 'Atlantic/South_Georgia',
            'GT': 'America/Guatemala',
            'GU': 'Pacific/Guam',
            'GW': 'Africa/Bissau',
            'GY': 'America/Guyana',
            'HK': 'Asia/Hong_Kong',
            'HM': 'Antarctica/Casey', 
            'HN': 'America/Tegucigalpa',
            'HR': 'Europe/Zagreb',
            'HT': 'America/Port-au-Prince',
            'HU': 'Europe/Budapest',
            'ID': 'Asia/Jakarta',
            'ID': 'Asia/Jayapura',
            'ID': 'Asia/Makassar',
            'IE': 'Europe/Dublin',
            'IL': 'Asia/Jerusalem',
            'IM': 'Europe/Isle_of_Man',
            'IN': 'Asia/Kolkata',
            'IO': 'Indian/Chagos',
            'IQ': 'Asia/Baghdad',
            'IR': 'Asia/Tehran',
            'IS': 'Atlantic/Reykjavik',
            'IT': 'Europe/Rome',
            'JE': 'Europe/Jersey',
            'JM': 'America/Jamaica',
            'JO': 'Asia/Amman',
            'JP': 'Asia/Tokyo',
            'KE': 'Africa/Nairobi',
            'KG': 'Asia/Bishkek',
            'KH': 'Asia/Phnom_Penh',
            'KI': 'Pacific/Enderbury',
            'KI': 'Pacific/Kiritimati',
            'KI': 'Pacific/Tarawa',
            'KM': 'Africa/Comoro',
            'KN': 'America/St_Kitts',
            'KP': 'Asia/Pyongyang',
            'KR': 'Asia/Seoul',
            'KW': 'Asia/Kuwait',
            'KY': 'America/Cayman',
            'KZ': 'Asia/Almaty',
            'KZ': 'Asia/Aqtau',
            'KZ': 'Asia/Aqtobe',
            'KZ': 'Asia/Astana',
            'KZ': 'Asia/Oral',
            'LA': 'Asia/Vientiane',
            'LB': 'Asia/Beirut',
            'LC': 'America/St_Lucia',
            'LI': 'Europe/Vaduz',
            'LK': 'Asia/Colombo',
            'LR': 'Africa/Monrovia',
            'LS': 'Africa/Maseru',
            'LT': 'Europe/Vilnius',
            'LU': 'Europe/Luxembourg',
            'LV': 'Europe/Riga',
            'LY': 'Africa/Tripoli',
            'MA': 'Africa/Casablanca',
            'MC': 'Europe/Monaco',
            'MD': 'Europe/Chisinau',
            'ME': 'Europe/Podgorica',
            'MF': 'America/St_Martin',
            'MG': 'Africa/Antananarivo',
            'MH': 'Pacific/Kwajalein',
            'MH': 'Pacific/Majuro',
            'MK': 'Europe/Skopje',
            'ML': 'Africa/Bamako',
            'MM': 'Asia/Yangon',
            'MN': 'Asia/Choibalsan',
            'MN': 'Asia/Hovd',
            'MN': 'Asia/Ulaanbaatar',
            'MO': 'Asia/Macau',
            'MP': 'Pacific/Saipan',
            'MQ': 'America/Martinique',
            'MR': 'Africa/Nouakchott',
            'MS': 'America/Montserrat',
            'MT': 'Europe/Malta',
            'MU': 'Indian/Mauritius',
            'MV': 'Indian/Maldives',
            'MW': 'Africa/Blantyre',
            'MX': 'America/Bahia_Banderas',
            'MX': 'America/Cancun',
            'MX': 'America/Chihuahua',
            'MX': 'America/Hermosillo',
            'MX': 'America/Matamoros',
            'MX': 'America/Mazatlan',
            'MX': 'America/Merida',
            'MX': 'America/Mexico_City',
            'MX': 'America/Monterrey',
            'MX': 'America/Ojinaga',
            'MX': 'America/Tijuana',
            'MY': 'Asia/Kuala_Lumpur',
            'MY': 'Asia/Kuching',
            'MZ': 'Africa/Maputo',
            'NA': 'Africa/Windhoek',
            'NC': 'Pacific/Noumea',
            'NE': 'Africa/Niamey',
            'NF': 'Pacific/Norfolk',
            'NG': 'Africa/Lagos',
            'NI': 'America/Managua',
            'NL': 'Europe/Amsterdam',
            'NO': 'Europe/Oslo',
            'NP': 'Asia/Kathmandu',
            'NR': 'Pacific/Nauru',
            'NU': 'Pacific/Niue',
            'NZ': 'Pacific/Auckland',
            'NZ': 'Pacific/Chatham',
            'OM': 'Asia/Muscat',
            'PA': 'America/Panama',
            'PE': 'America/Lima',
            'PF': 'Pacific/Gambier',
            'PF': 'Pacific/Marquesas',
            'PF': 'Pacific/Tahiti',
            'PG': 'Pacific/Bougainville',
            'PG': 'Pacific/Port_Moresby',
            'PH': 'Asia/Manila',
            'PK': 'Asia/Karachi',
            'PL': 'Europe/Warsaw',
            'PM': 'America/Miquelon',
            'PN': 'Pacific/Pitcairn',
            'PR': 'America/Puerto_Rico',
            'PS': 'Asia/Gaza',
            'PS': 'Asia/Hebron',
            'PT': 'Atlantic/Azores',
            'PT': 'Atlantic/Madeira',
            'PT': 'Europe/Lisbon',
            'PW': 'Pacific/Palau',
            'PY': 'America/Asuncion',
            'QA': 'Asia/Qatar',
            'RE': 'Indian/Reunion',
            'RO': 'Europe/Bucharest',
            'RS': 'Europe/Belgrade',
            'RU': 'Asia/Anadyr',
            'RU': 'Asia/Barnaul',
            'RU': 'Asia/Chita',
            'RU': 'Asia/Irkutsk',
            'RU': 'Asia/Kamchatka',
            'RU': 'Asia/Khandyga',
            'RU': 'Asia/Krasnoyarsk',
            'RU': 'Asia/Magadan',
            'RU': 'Asia/Novokuznetsk',
            'RU': 'Asia/Novosibirsk',
            'RU': 'Asia/Omsk',
            'RU': 'Asia/Sakhalin',
            'RU': 'Asia/Srednekolymsk',
            'RU': 'Asia/Tomsk',
            'RU': 'Asia/Ust-Nera',
            'RU': 'Asia/Vladivostok',
            'RU': 'Asia/Yakutsk',
            'RU': 'Asia/Yekaterinburg',
            'RU': 'Europe/Kaliningrad',
            'RU': 'Europe/Kirov',
            'RU': 'Europe/Moscow',
            'RU': 'Europe/Samara',
            'RU': 'Europe/Simferopol',
            'RU': 'Europe/Ulyanovsk',
            'RW': 'Africa/Kigali',
            'SA': 'Asia/Riyadh',
            'SB': 'Pacific/Guadalcanal',
            'SC': 'Indian/Mahe',
            'SD': 'Africa/Khartoum',
            'SE': 'Europe/Stockholm',
            'SG': 'Asia/Singapore',
            'SH': 'Atlantic/St_Helena',
            'SI': 'Europe/Ljubljana',
            'SJ': 'Arctic/Longyearbyen',
            'SK': 'Europe/Bratislava',
            'SL': 'Africa/Freetown',
            'SM': 'Europe/San_Marino',
            'SN': 'Africa/Dakar',
            'SO': 'Africa/Mogadishu',
            'SR': 'America/Paramaribo',
            'SS': 'Africa/Juba',
            'ST': 'Africa/Sao_Tome',
            'SV': 'America/El_Salvador',
            'SX': 'America/Lower_Princes',
            'SY': 'Asia/Damascus',
            'SZ': 'Africa/Mbabane',
            'TC': 'America/Grand_Turk',
            'TD': 'Africa/Ndjamena',
            'TF': 'Indian/Kerguelen',
            'TG': 'Africa/Lome',
            'TH': 'Asia/Bangkok',
            'TJ': 'Asia/Dushanbe',
            'TK': 'Pacific/Fakaofo',
            'TL': 'Asia/Dili',
            'TM': 'Asia/Ashgabat',
            'TN': 'Africa/Tunis',
            'TO': 'Pacific/Tongatapu',
            'TR': 'Europe/Istanbul',
            'TT': 'America/Port_of_Spain',
            'TV': 'Pacific/Funafuti',
            'TW': 'Asia/Taipei',
            'TZ': 'Africa/Dar_es_Salaam',
            'UA': 'Europe/Kiev',
            'UA': 'Europe/Simferopol',
            'UA': 'Europe/Uzhgorod',
            'UA': 'Europe/Zaporozhye',
            'UG': 'Africa/Kampala',
            'UM': 'Pacific/Midway',
            'UM': 'Pacific/Wake',
            'US': 'America/Adak',
            'US': 'America/Anchorage',
            'US': 'America/Boise',
            'US': 'America/Chicago',
            'US': 'America/Denver',
            'US': 'America/Detroit',
            'US': 'America/Indiana/Indianapolis',
            'US': 'America/ Indiana/Knox',
            'US': 'America/Indiana/Marengo',
            'US': 'America/Indiana/Petersburg',
            'US': 'America/Indiana/Tell_City',
            'US': 'America/Indiana/Vevay',
            'US': 'America/Indiana/Vincennes',
            'US': 'America/Indiana/Winamac',
            'US': 'America/Juneau', 
            'US': 'America/Kentucky/Louisville',
            'US': 'America/Kentucky/Monticello',
            'US': 'America/Los_Angeles',
            'US': 'America/Menominee',
            'US': 'America/Metlakatla',
            'US': 'America/New_York',
            'US': 'America/Nome',
            'US': 'America/North_Dakota/Beulah',
            'US': 'America/North_Dakota/Center',
            'US': 'America/North_Dakota/New_Salem',
            'US': 'America/Phoenix',
            'US': 'America/Sitka',  
            'US': 'America/Yakutat',
            'US': 'Pacific/Honolulu',
            'US': 'Pacific/Johnston',
            'US': 'Pacific/Midway',
            'US': 'Pacific/Wake',
            'UY': 'America/Montevideo',
            'UZ': 'Asia/Samarkand',
            'UZ': 'Asia/Tashkent',
            'VA': 'Europe/Vatican',
            'VC': 'America/St_Vincent',
            'VE': 'America/Caracas',
            'VG': 'America/Tortola',
                    'VI': 'America/St_Thomas',  
                    'VN': 'Asia/Ho_Chi_Minh',
                    'VU': 'Pacific/Efate',
                    'WF': 'Pacific/Wallis',
                    'WS': 'Pacific/Apia',
                    'YE': 'Asia/Aden',
                    'YT': 'Indian/Mayotte',
                    'ZA': 'Africa/Johannesburg',
                    'ZM': 'Africa/Lusaka',
                    'ZW': 'Africa/Harare'
                }
        
        # Some countries have multiple time zones - this handles special cases
        self.multi_timezone_countries = {
                'US': {
                    'EST': 'America/New_York',
                    'CST': 'America/Chicago',
                    'MST': 'America/Denver',
                    'PST': 'America/Los_Angeles',
                    'AKST': 'America/Anchorage',
                    'HST': 'Pacific/Honolulu',
                    'AST': 'America/Puerto_Rico',  # Atlantic (Puerto Rico, US Virgin Islands)
                    'SST': 'Pacific/Pago_Pago',    # Samoa Time (American Samoa)
                    'ChST': 'Pacific/Guam'         # Chamorro Time (Guam)
                },
                'RU': {
                    'MSK-1': 'Europe/Kaliningrad',
                    'MSK': 'Europe/Moscow',
                    'MSK+1': 'Europe/Samara',
                    'MSK+2': 'Asia/Yekaterinburg',
                    'MSK+3': 'Asia/Omsk',
                    'MSK+4': 'Asia/Krasnoyarsk',
                    'MSK+5': 'Asia/Irkutsk',
                    'MSK+6': 'Asia/Yakutsk',
                    'MSK+7': 'Asia/Vladivostok',
                    'MSK+8': 'Asia/Magadan',
                    'MSK+9': 'Asia/Kamchatka'
                },
                'CA': {
                    'NST': 'America/St_Johns',     # Newfoundland
                    'AST': 'America/Halifax',      # Atlantic
                    'EST': 'America/Toronto',      # Eastern
                    'CST': 'America/Winnipeg',     # Central
                    'MST': 'America/Edmonton',     # Mountain
                    'PST': 'America/Vancouver'     # Pacific
                },
                'AU': {
                    'AEST': 'Australia/Sydney',    # Eastern
                    'ACST': 'Australia/Adelaide',  # Central
                    'AWST': 'Australia/Perth'      # Western
                },
                'BR': {
                    'BRT': 'America/Sao_Paulo',    # Brasília Time
                    'AMT': 'America/Manaus',       # Amazon Time
                    'FNT': 'America/Noronha',      # Fernando de Noronha Time
                    'ACT': 'America/Rio_Branco'    # Acre Time
                },
                'MX': {
                    'EST': 'America/Cancun',       # Eastern
                    'CST': 'America/Mexico_City',  # Central
                    'MST': 'America/Chihuahua',    # Mountain
                    'PST': 'America/Tijuana'       # Pacific
                },
                'ID': {
                    'WIB': 'Asia/Jakarta',         # Western Indonesian Time
                    'WITA': 'Asia/Makassar',       # Central Indonesian Time
                    'WIT': 'Asia/Jayapura'         # Eastern Indonesian Time
                },
                'CD': {
                    'WAT': 'Africa/Kinshasa',      # Western Congo
                    'CAT': 'Africa/Lubumbashi'     # Eastern Congo
                },
                'KZ': {
                    'WEST': 'Asia/Aqtau',          # Western Kazakhstan
                    'EAST': 'Asia/Almaty'          # Eastern Kazakhstan
                },
                'MN': {
                    'HOVT': 'Asia/Hovd',           # Western Mongolia
                    'ULAT': 'Asia/Ulaanbaatar'     # Central/Eastern Mongolia
                },
                'KI': {
                    'GILT': 'Pacific/Tarawa',      # Gilbert Islands
                    'PHOT': 'Pacific/Enderbury',   # Phoenix Islands
                    'LINT': 'Pacific/Kiritimati'   # Line Islands
                },
                'FR': {
                    'CET': 'Europe/Paris',         # Metropolitan France
                    'RET': 'Indian/Reunion',       # Réunion
                    'NCT': 'Pacific/Noumea',       # New Caledonia
                    'WFT': 'Pacific/Wallis',       # Wallis and Futuna
                    'TAHT': 'Pacific/Tahiti',      # French Polynesia (Tahiti)
                    'MART': 'Pacific/Marquesas',   # French Polynesia (Marquesas)
                    'GFT': 'America/Cayenne',      # French Guiana
                    'PMST': 'America/Miquelon',    # Saint Pierre and Miquelon
                    'AST': 'America/Martinique'    # French Antilles (Martinique, Guadeloupe)
                },
                'GB': {
                    'GMT': 'Europe/London',        # United Kingdom
                    'BST': 'Atlantic/Bermuda',     # Bermuda
                    'FKST': 'Atlantic/Stanley',    # Falkland Islands
                    'GST': 'Atlantic/South_Georgia', # South Georgia
                    'AEDT': 'Indian/Chagos',       # British Indian Ocean Territory
                    'GBT': 'Africa/Gibraltar',     # Gibraltar
                    'AWST': 'Indian/Christmas',    # Christmas Island
                    'ACST': 'Indian/Cocos'         # Cocos Islands
                },
                'ES': {
                    'CET': 'Europe/Madrid',        # Mainland Spain
                    'WET': 'Atlantic/Canary'       # Canary Islands
                },
                'PT': {
                    'WET': 'Europe/Lisbon',        # Mainland Portugal
                    'AZOT': 'Atlantic/Azores'      # Azores
                },
                'CL': {
                    'CLT': 'America/Santiago',     # Continental Chile
                    'EAST': 'Pacific/Easter'       # Easter Island
                },
                'EC': {
                    'ECT': 'America/Guayaquil',    # Mainland Ecuador
                    'GALT': 'Pacific/Galapagos'    # Galápagos Islands
                },
                'FM': {
                    'CHUT': 'Pacific/Chuuk',       # Chuuk/Yap
                    'PONT': 'Pacific/Pohnpei'      # Pohnpei/Kosrae
                },
                'PG': {
                    'PGT': 'Pacific/Port_Moresby', # Main country
                    'BST': 'Pacific/Bougainville'  # Bougainville
                },
                'AQ': {
                    'NZST': 'Antarctica/McMurdo',  # New Zealand Time (McMurdo, Scott Base)
                    'CLST': 'Antarctica/Palmer',   # Chile Time (Palmer)
                    'AEDT': 'Antarctica/Casey',    # Australian Eastern Time (Casey)
                    'DAVT': 'Antarctica/Davis',    # Davis Time
                    'VOST': 'Antarctica/Vostok',   # Vostok Time
                    'MAWT': 'Antarctica/Mawson',   # Mawson Time
                    'ROTT': 'Antarctica/Rothera',  # Rothera Time
                    'SYOT': 'Antarctica/Syowa',    # Syowa Time
                    'TROL': 'Antarctica/Troll'     # Troll Time
                }
            }
        
    
    def get_timezone(self, country_code: str, region: str = None) -> str:
        """Get time zone for a country code and optional region"""
        country_code = country_code.upper() if country_code else 'US'  # Default to US
        
        # Check for specific region in multi-timezone country
        if country_code in self.multi_timezone_countries and region:
            region_timezones = self.multi_timezone_countries[country_code]
            if region in region_timezones:
                return region_timezones[region]
        
        # Return the primary time zone for the country
        return self.country_to_timezone.get(country_code, 'UTC')
    
    def convert_to_local_time(self, utc_time: datetime, country_code: str, region: str = None) -> datetime:
        """Convert UTC time to local time for given country/region"""
        import pytz
        from datetime_conversion_handler import convert_timestamp_to_datetime
        
        # Convert input to a consistent UTC datetime
        try:
            # Use our robust conversion method
            utc_time = convert_timestamp_to_datetime(utc_time)
            
            # Get timezone string
            timezone_str = self.get_timezone(country_code, region)
            local_tz = pytz.timezone(timezone_str)
            
            # Convert to local time
            return utc_time.astimezone(local_tz)
        
        except Exception as e:
            logger.error(f"Error converting to local time: {e}")
            # Fallback to UTC if conversion fails
            return utc_time

    def is_business_hours(self, time: datetime, country_code: str, 
                        business_start: int = 9, business_end: int = 17) -> bool:
        """
        Check if time is within business hours for given country
        
        Args:
            time: Input time (can be various datetime-like types)
            country_code: ISO country code
            business_start: Start of business hours (default 9 AM)
            business_end: End of business hours (default 5 PM)
        
        Returns:
            Boolean indicating if time is within business hours
        """
        import pytz
        from datetime_conversion_handler import convert_timestamp_to_datetime
        
        try:
            # Convert input to a consistent UTC datetime
            utc_time = convert_timestamp_to_datetime(time)
            
            # Get local time for the country
            timezone_str = self.get_timezone(country_code)
            local_tz = pytz.timezone(timezone_str)
            
            # Convert to local time
            local_time = utc_time.astimezone(local_tz)
            
            # Check business hours
            local_hour = local_time.hour
            weekday = local_time.weekday()
            
            # Business hours check (weekdays only)
            return (
                business_start <= local_hour < business_end and 
                weekday < 5  # Monday to Friday (0-4)
            )
        
        except Exception as e:
            logger.error(f"Error checking business hours: {e}")
            
            # Fallback to default business hours check using UTC time
            try:
                utc_time = convert_timestamp_to_datetime(time)
                local_hour = utc_time.hour
                weekday = utc_time.weekday()
                
                return (
                    business_start <= local_hour < business_end and 
                    weekday < 5
                )
            except Exception:
                # Absolute fallback
                current_time = datetime.now()
                return (
                    business_start <= current_time.hour < business_end and 
                    current_time.weekday() < 5
                )

    def get_business_hours(self, country_code: str) -> Tuple[int, int]:
        """Get standard business hours for a country"""
        business_hours = {
            'US': (9, 17),  # 9 AM to 5 PM
            'GB': (9, 17),
            'JP': (9, 18),  # 9 AM to 6 PM
            'AE': (8, 16),  # 8 AM to 4 PM (Middle East often has different work weeks)
            # Add more countries as needed
        }
        return business_hours.get(country_code, (9, 17))  # Default to 9-5
    
class ISO20022Parser:
    def __init__(self, config_reader=None):
        self.validator = ISO20022MessageValidator()
        self.channel_mapper = MessageChannelMapper()
        self.config_reader = config_reader 

        if self.config_reader is None:
            try:
                from configreader import ConfigReader
                self.config_reader = ConfigReader('./config.properties')
            except Exception as e:
                #logger.warning(f"Could not initialize default ConfigReader: {e}")
                self.config_reader = None
    
    def safe_strip(self, value: Optional[str]) -> str:
            """Safely strip a string value that might be None"""
            if value is None:
                return ''
            return str(value).strip()

    def safe_get(self, dict_obj: Optional[Dict], key: str, default: Any = '') -> Any:
        """Safely get a value from a dictionary that might be None"""
        if dict_obj is None:
            return default
        return dict_obj.get(key, default)
    
    def _ensure_namespaces(self, xml_content: str) -> str:
        """Ensure all required namespaces are present in the XML"""
        try:
            # Define required namespaces
            required_namespaces = {
                'pain': 'urn:iso:std:iso:20022:tech:xsd:pain.001.001.09',
                'pacs': 'urn:iso:std:iso:20022:tech:xsd:pacs.008.001.08',
                'camt': 'urn:iso:std:iso:20022:tech:xsd:camt.054.001.08'
            }
            
            # Check if this is a Document element without proper namespace
            for prefix, uri in required_namespaces.items():
                if f"<{prefix}:Document" in xml_content and f"xmlns:{prefix}" not in xml_content:
                    # Find the Document tag
                    doc_pos = xml_content.find(f"<{prefix}:Document")
                    if doc_pos != -1:
                        # Find the end of the tag
                        close_pos = xml_content.find('>', doc_pos)
                        if close_pos != -1:
                            # Insert namespace declaration
                            xml_content = (
                                xml_content[:close_pos] +
                                f' xmlns:{prefix}="{uri}"' +
                                xml_content[close_pos:]
                            )

            return xml_content
            
        except Exception as e:
            logger.error(f"Error ensuring namespaces: {str(e)}")
            return xml_content  # Return original content if something goes wrong

    def _create_default_parsed_message(self) -> Dict:
        """Create default parsed message structure"""
        return {
            'header': ISO20022Header(
                message_id='UNKNOWN',
                creation_datetime=datetime.now(),
                number_of_transactions=1,
                control_sum=None,
                initiating_party={},
                message_type=ISO20022MessageType.PAIN_001
            ),
            'content': {'payment_informations': []},
            'payment_channel': PaymentChannel.WIRE,
            'raw_xml': ''
        }

    def _extract_namespaces(self, root: ET.Element) -> Dict[str, str]:
        """Extract all namespaces from XML document"""
        namespaces = {}
        
        # Get namespaces from root element
        for key, value in root.attrib.items():
            if key.startswith('{http://www.w3.org/2000/xmlns/}'):
                prefix = key.split('}')[1]
                namespaces[prefix] = value
                
        return namespaces

    def _create_namespace_mapping(self, message_type: ISO20022MessageType, doc_namespaces: Dict[str, str]) -> Dict[str, str]:
        """Create namespace mapping based on message type and document namespaces"""
        standard_mapping = {
            ISO20022MessageType.PAIN_001: {'pain': self.validator.namespaces['pain']},
            ISO20022MessageType.PACS_008: {'pacs': self.validator.namespaces['pacs']},
            ISO20022MessageType.CAMT_054: {'camt': self.validator.namespaces['camt']}
        }
        
        # Get standard mapping for message type
        ns_map = standard_mapping.get(message_type, {})
        
        # Add any additional namespaces from document
        for prefix, uri in doc_namespaces.items():
            if prefix not in ns_map and uri in self.validator.namespaces.values():
                ns_map[prefix] = uri
                
        return ns_map

    def parse_message(self, xml_content: str) -> Dict[str, Any]:
        """Parse ISO20022 message with improved namespace handling"""
        try:
            # Fix mixed namespace issues first
            xml_content = self._fix_mixed_namespaces(xml_content)
            
            try:
                root = ET.fromstring(xml_content)
            except ET.ParseError as e:
                logger.error(f"XML parsing error: {str(e)}")
                logger.debug(f"XML content snippet: {xml_content[:200]}...")
                return self._create_default_parsed_message()

            # Get all namespaces used in the document
            namespaces = self._extract_namespaces(root)
            
            # Detect message type using available namespaces
            message_type = self.validator.detect_message_type(root)
            
            # Create namespace mapping
            ns = self._create_namespace_mapping(message_type, namespaces)
            
            # Validate structure
            self.validator.validate_structure(root, message_type)
            
            # Determine payment channel
            payment_channel = self.channel_mapper.determine_channel(message_type, xml_content)
            
            # Parse based on message type
            if message_type == ISO20022MessageType.PAIN_001:
                content = self._parse_pain001(root, ns)
            elif message_type == ISO20022MessageType.PACS_008:
                content = self._parse_pacs008(root, ns)
            else:  # CAMT.054
                content = self._parse_camt054(root, ns)
            
            # Create header
            header = self._create_header(root, message_type, ns)
            
            return {
                'header': header,
                'content': content,
                'payment_channel': payment_channel,
                'raw_xml': xml_content
            }
            
        except Exception as e:
            logger.error(f"Error parsing message: {str(e)}")
            return self._create_default_parsed_message()
    
    def _fix_namespace_declarations(self, xml_content: str) -> str:
        """Fix mixed namespace issues in the XML content"""
        try:
            # Define standard namespace URIs
            standard_ns = {
                'pain': 'urn:iso:std:iso:20022:tech:xsd:pain.001.001.09',
                'pacs': 'urn:iso:std:iso:20022:tech:xsd:pacs.008.001.08',
                'camt': 'urn:iso:std:iso:20022:tech:xsd:camt.054.001.08'
            }
            
            # Replace any nsX: prefix used with the standard namespace
            if 'xmlns:ns' in xml_content:
                for std_prefix, uri in standard_ns.items():
                    if uri in xml_content:
                        # Find all nsX prefixes used with this URI
                        ns_pattern = f'xmlns:ns\\d+="{uri}"'
                        matches = re.finditer(ns_pattern, xml_content)
                        for match in matches:
                            ns_decl = match.group(0)
                            ns_prefix = ns_decl.split('=')[0].split(':')[1]
                            # Replace all occurrences of this prefix with standard prefix
                            xml_content = xml_content.replace(f'{ns_prefix}:', f'{std_prefix}:')
                            # Update namespace declaration
                            xml_content = xml_content.replace(ns_decl, f'xmlns:{std_prefix}="{uri}"')

            # Ensure all required namespaces are declared
            for prefix, uri in standard_ns.items():
                if f'{prefix}:' in xml_content and f'xmlns:{prefix}' not in xml_content:
                    insert_pos = xml_content.find('>')
                    if insert_pos > 0:
                        xml_content = (
                            xml_content[:insert_pos] +
                            f' xmlns:{prefix}="{uri}"' +
                            xml_content[insert_pos:]
                        )

            return xml_content
            
        except Exception as e:
            logger.error(f"Error fixing mixed namespaces: {str(e)}")
            return xml_content

    def _create_default_parsed_message(self) -> Dict:
        """Create default parsed message structure"""
        return {
            'header': ISO20022Header(
                message_id='UNKNOWN',
                creation_datetime=datetime.now(),
                number_of_transactions=1,
                control_sum=None,
                initiating_party={},
                message_type=ISO20022MessageType.PAIN_001
            ),
            'content': {'payment_informations': []},
            'payment_channel': PaymentChannel.WIRE,
            'raw_xml': ''
        }
    
    @staticmethod
    def _parse_datetime(datetime_val: Optional[Union[str, datetime]]) -> Optional[datetime]:
        """
        Helper function to parse datetime strings with various formats
        Also handles cases where input is already a datetime object
        """
        import pytz
        
        if datetime_val is None:
            return None
            
        # If already a datetime object, return it as is
        if isinstance(datetime_val, datetime) and datetime_val.tzinfo is not None:
            return datetime_val.astimezone(pytz.UTC)
            
        # If it's a pandas Timestamp, convert to datetime
        if isinstance(datetime_val, pd.Timestamp):
            dt = datetime_val.to_pydatetime()
            if dt.tzinfo is None:
                return pytz.UTC.localize(dt)
            return dt.astimezone(pytz.UTC)
        
        # If already a datetime object without timezone
        if isinstance(datetime_val, datetime):
            return pytz.UTC.localize(datetime_val)
            
        if not isinstance(datetime_val, str):
            #logger.warning(f"Unexpected datetime type: {type(datetime_val)}")
            return pytz.UTC.localize(datetime.now())
        
            
        formats = [
            '%Y-%m-%dT%H:%M:%SZ',     # Format with Z
            '%Y-%m-%dT%H:%M:%S.%fZ',  # Format with microseconds and Z
            '%Y-%m-%dT%H:%M:%S',      # Without Z
            '%Y-%m-%dT%H:%M:%S.%f',   # With microseconds
            '%Y-%m-%d %H:%M:%S.%f',   # Alternative format with microseconds
            '%Y-%m-%d %H:%M:%S',      # Alternative format without microseconds
            '%Y-%m-%d'                # Date only format
        ]
        
        for fmt in formats:
            try:
                parsed_time = datetime.strptime(datetime_val, fmt)
                # Localize to UTC if no timezone info
                if parsed_time.tzinfo is None:
                    return pytz.UTC.localize(parsed_time)
                return parsed_time.astimezone(pytz.UTC)
            except ValueError:
                continue
                
         #logger.warning(f"Could not parse datetime string: {datetime_val}")
        return pytz.UTC.localize(datetime.now())  # Return current time as fallback, localized to UTC
    
    def _create_header(self, root: ET.Element, message_type: ISO20022MessageType, ns: Dict[str, str]) -> ISO20022Header:
        """Create ISO20022Header object based on message type"""
        try:
            ns_prefix = list(ns.keys())[0]
            
            if message_type == ISO20022MessageType.CAMT_054:
                grp_hdr = root.find(f'.//{ns_prefix}:BkToCstmrDbtCdtNtfctn/{ns_prefix}:GrpHdr', ns)
            elif message_type == ISO20022MessageType.PAIN_001:
                grp_hdr = root.find(f'.//{ns_prefix}:CstmrCdtTrfInitn/{ns_prefix}:GrpHdr', ns)
            else:  # PACS.008
                grp_hdr = root.find(f'.//{ns_prefix}:FIToFICstmrCdtTrf/{ns_prefix}:GrpHdr', ns)
            
            if grp_hdr is None:
                raise ValueError("Group Header not found")
            
            msg_id = grp_hdr.find(f'.//{ns_prefix}:MsgId', ns)
            creation_dt = grp_hdr.find(f'.//{ns_prefix}:CreDtTm', ns)
            nb_of_txs = grp_hdr.find(f'.//{ns_prefix}:NbOfTxs', ns)
            ctrl_sum = grp_hdr.find(f'.//{ns_prefix}:CtrlSum', ns)

            # Use _parse_datetime helper for creation datetime
            creation_datetime = self._parse_datetime(creation_dt.text if creation_dt is not None else None)

            return ISO20022Header(
                message_id=msg_id.text if msg_id is not None else None,
                creation_datetime=creation_datetime,
                number_of_transactions=int(nb_of_txs.text) if nb_of_txs is not None else 1,
                control_sum=float(ctrl_sum.text) if ctrl_sum is not None else None,
                initiating_party={},  # Simplified for now
                message_type=message_type
            )
        except Exception as e:
            logger.error(f"Error creating header: {str(e)}")
            raise

    def _convert_to_transaction_format(self, parsed_message: Dict) -> Dict:
        """Convert parsed ISO20022 message to standard transaction format with improved datetime handling"""
        try:
            header = parsed_message['header']
            content = parsed_message['content']
            
            # Handle creation_datetime safely
            if isinstance(header.creation_datetime, (datetime, pd.Timestamp)):
                creation_date = header.creation_datetime
            else:
                creation_date = self._parse_datetime(header.creation_datetime)

            # Determine high-risk countries
            high_risk_countries = set()
            if self.config_reader:
                # Try to read from config, with a default list
                high_risk_countries = set(
                    self.config_reader.get_property(
                        'global.high_risk_countries', 
                        'AF,KP,IR,MM,SS,SY,VE,YE'
                    ).split(',')
                )
            else:
                # Fallback to hardcoded list if no config reader
                high_risk_countries = {'AF', 'KP', 'IR', 'MM', 'SS', 'SY', 'VE', 'YE'}
            
            # Initialize transaction with required fields and safe defaults
            transaction = {
                'msg_id': header.message_id,
                'creation_date': creation_date,  # Now handled safely
                'channel': parsed_message['payment_channel'].value,
                'pmt_info':None,
                'amount': 0.0,
                'currency': 'USD',
                'transaction_type': 'UNKNOWN',
                'debtor_account': None,
                'debtor_account_routing_number':None,
                'debtor_agent_BIC':None,
                'creditor_account':None,
                'creditor_account_routing_number':None, 
                'creditor_name':None, 
                'credit_agent_BIC':None,
                'purpose_code': None, 
                'remittance_info':None, 
                'mandate_id':None,
                'is_cross_border': False,
                'involves_high_risk_country': False,
                'booking_date': None,
                'value_date': None,
                'status': None
            }

            try:
                # Extract information based on message type
                if header.message_type == ISO20022MessageType.PAIN_001:
                    if 'payment_informations' in content and content['payment_informations']:
                        pmt_info = content['payment_informations'][0]
                        if 'transactions' in pmt_info and pmt_info['transactions']:
                            tx = pmt_info['transactions'][0]
                            self._update_transaction_pain001(transaction, pmt_info, tx, high_risk_countries)
                            
                elif header.message_type == ISO20022MessageType.PACS_008:
                    if 'credit_transfers' in content and content['credit_transfers']:
                        tx = content['credit_transfers'][0]
                        self._update_transaction_pacs008(transaction, tx, high_risk_countries)
                        
                elif header.message_type == ISO20022MessageType.CAMT_054:
                    if 'notifications' in content and content['notifications']:
                        notification = content['notifications'][0]
                        if 'entries' in notification and notification['entries']:
                            entry = notification['entries'][0]
                            self._update_transaction_camt054(transaction, notification, entry, high_risk_countries)
            except Exception as e:
                logger.error(f"Error updating transaction details: {str(e)}")
                # Continue with default values if update fails
            
            return transaction
            
        except Exception as e:
            logger.error(f"Error converting transaction format: {str(e)}")
            raise

    def parse_message(self, xml_content: str) -> Dict:
        """Parse an ISO20022 message"""
        try:
            root = ET.fromstring(xml_content)
            message_type = self.validator.detect_message_type(root)
            
            # Validate the message structure
            self.validator.validate_structure(root, message_type)
            
            # Get the appropriate namespace
            if message_type == ISO20022MessageType.PAIN_001:
                ns = {'pain': self.validator.namespaces['pain']}
            elif message_type == ISO20022MessageType.PACS_008:
                ns = {'pacs': self.validator.namespaces['pacs']}
            else:  # CAMT.054
                ns = {'camt': self.validator.namespaces['camt']}
            
            # Determine payment channel
            payment_channel = self.channel_mapper.determine_channel(
                message_type, 
                xml_content
            )
            
            # Parse based on message type
            if message_type == ISO20022MessageType.PAIN_001:
                content = self._parse_pain001(root, ns)
            elif message_type == ISO20022MessageType.PACS_008:
                content = self._parse_pacs008(root, ns)
            else:  # CAMT.054
                content = self._parse_camt054(root, ns)
            
            # Create header using _parse_datetime for date handling
            header = self._create_header(root, message_type, ns)
            
            return {
                'header': header,
                'content': content,
                'payment_channel': payment_channel,
                'raw_xml': xml_content
            }
            
        except Exception as e:
            logger.error(f"Error parsing message: {str(e)}")
            raise
    
    def parse_message(self, xml_content: str) -> Dict:
        """Parse an ISO20022 message"""
        try:
            root = ET.fromstring(xml_content)
            message_type = self.validator.detect_message_type(root)
            
            # Validate the message structure
            self.validator.validate_structure(root, message_type)
            
            # Get the appropriate namespace
            if message_type == ISO20022MessageType.PAIN_001:
                ns = {'pain': self.validator.namespaces['pain']}
            elif message_type == ISO20022MessageType.PACS_008:
                ns = {'pacs': self.validator.namespaces['pacs']}
            else:  # CAMT.054
                ns = {'camt': self.validator.namespaces['camt']}
            
            # Determine payment channel
            payment_channel = self.channel_mapper.determine_channel(
                message_type, 
                xml_content
            )
            
            # Parse based on message type
            if message_type == ISO20022MessageType.PAIN_001:
                content = self._parse_pain001(root, ns)
            elif message_type == ISO20022MessageType.PACS_008:
                content = self._parse_pacs008(root, ns)
            else:  # CAMT.054
                content = self._parse_camt054(root, ns)
            
            # Create header using _parse_datetime for date handling
            header = self._create_header(root, message_type, ns)
            
            return {
                'header': header,
                'content': content,
                'payment_channel': payment_channel,
                'raw_xml': xml_content
            }
            
        except Exception as e:
            logger.error(f"Error parsing message: {str(e)}")
            raise
    
    def _parse_pain001(self, root: ET.Element, ns: Dict[str, str]) -> Dict:
        # Use the default namespace from the XML
        if ns is None or not ns:
            ns = {'pain': 'urn:iso:std:iso:20022:tech:xsd:pain.001.001.09'}
        
        payment_infos = []
        xml_content = ET.tostring(root, encoding='unicode', method='xml')
        
        

        try:
            for pmt_inf in root.findall('.//pain:PmtInf', ns):
                try:
                    # Initialize variables with default values
                    debtor_agent_bic = None
                    routing_number = None
                    
                    # Create payment info dictionary with safe defaults
                    info = {
                        'document': xml_content[:65000] if len(xml_content) > 65000 else xml_content,
                        'payment_info_id': self._safe_find_text(pmt_inf, './/pain:PmtInfId', ns) or '',
                        'payment_method': self._safe_find_text(pmt_inf, './/pain:PmtMtd', ns) or '',
                        'debtor_routing_number': routing_number or '',
                        'debtor_agent_bic': self._safe_find_text(pmt_inf.find('.//pain:DbtrAgt/pain:FinInstnId', ns), 'pain:BICFI', ns) or '',
                        'debtor': self._parse_party_info(pmt_inf.find('.//pain:Dbtr', ns), 'pain', ns) or {},
                        'debtor_account': self._parse_account_info(pmt_inf.find('.//pain:DbtrAcct', ns), 'pain', ns) or {},
                        'transactions': []
                    }
                    
                    # Process transactions
                    for tx_inf in pmt_inf.findall('.//pain:CdtTrfTxInf', ns):
                        try:
                            # Get amount element and safely extract value and currency
                            amt_elem = tx_inf.find('.//pain:Amt/pain:InstdAmt', ns)
                            amount = float(amt_elem.text) if amt_elem is not None else 0.0
                            currency = amt_elem.get('Ccy') if amt_elem is not None else None
                            
                            transaction = {
                                'instruction_id': self._safe_find_text(tx_inf, './/pain:PmtId/pain:InstrId', ns) or '',
                                'end_to_end_id': self._safe_find_text(tx_inf, './/pain:PmtId/pain:EndToEndId', ns) or '',
                                'amount': amount,
                                'currency': currency or 'USD',
                                'creditor': self._parse_party_info(tx_inf.find('.//pain:Cdtr', ns), 'pain', ns) or {},
                                'creditor_account': self._parse_account_info(tx_inf.find('.//pain:CdtrAcct', ns), 'pain', ns) or {},
                                'creditor_agent_bic':self._safe_find_text(pmt_inf.find('.//pain:CdtrAgt/pain:FinInstnId', ns), 'pain:BICFI', ns) or '',
                                'purpose': self._safe_find_text(tx_inf, './/pain:Purp/pain:Cd', ns) or '',
                                'remittance_info': self._safe_find_text(tx_inf, './/pain:RmtInf/pain:Ustrd', ns) or ''
                            }
                            
                            info['transactions'].append(transaction)
                        except Exception as tx_error:
                            logger.warning(f"Error processing transaction in PAIN.001: {str(tx_error)}")
                            continue
                    
                    payment_infos.append(info)
                    
                except Exception as pmt_error:
                    logger.warning(f"Error processing payment info in PAIN.001: {str(pmt_error)}")
                    continue
            
            return {'payment_informations': payment_infos}
            
        except Exception as e:
            logger.error(f"Error parsing PAIN.001 message: {str(e)}")
            # Return minimal valid structure instead of raising
            return {'payment_informations': []}
        
    def _parse_pacs008(self, root: ET.Element, ns: Dict[str, str]) -> Dict:
        """Parse PACS.008 message"""
        transactions = []
        xml_content = ET.tostring(root, encoding='unicode', method='xml')

        def extract_account_number_dbtr(tx_inf):
            """Extract debtor account number with multiple fallback methods"""
            # First, try IBAN
            iban = self._safe_find_text(tx_inf, './/pacs:DbtrAcct/pacs:Id/pacs:IBAN', ns)
            if iban:
                return iban
            
            # Next, try Othr (Other) identifier
            othr = tx_inf.find('.//pacs:DbtrAcct/pacs:Id/pacs:Othr', ns)
            if othr is not None:
                return othr.text
            
            return None

        def extract_account_number_cdtr(tx_inf):
            """Extract creditor account number with multiple fallback methods"""
            # First, try IBAN
            iban = self._safe_find_text(tx_inf, './/pacs:CdtrAcct/pacs:Id/pacs:IBAN', ns)
            if iban:
                return iban
            
            # Next, try Othr (Other) identifier
            othr = tx_inf.find('.//pacs:CdtrAcct/pacs:Id/pacs:Othr', ns)
            if othr is not None:
                return othr.text
            
            return None

        for tx_inf in root.findall('.//pacs:CdtTrfTxInf', ns):
            # Get debtor agent info
            dbtr_agt = tx_inf.find('.//pacs:DbtrAgt/pacs:FinInstnId', ns)
            dbtr_bic = None
            dbtr_routing = None
            
            if dbtr_agt is not None:
                dbtr_bic = dbtr_agt.findtext('pacs:BICFI', namespaces=ns)
                if not dbtr_bic:
                    # Try clearing system member ID if BIC not found
                    dbtr_clr = dbtr_agt.find('.//pacs:ClrSysMmbId/pacs:MmbId', ns)
                    dbtr_routing = dbtr_clr.text if dbtr_clr is not None else None
                else:
                    dbtr_routing = dbtr_bic

            # Get creditor agent info
            cdtr_agt = tx_inf.find('.//pacs:CdtrAgt/pacs:FinInstnId', ns)
            cdtr_bic = None
            cdtr_routing = None
            
            if cdtr_agt is not None:
                cdtr_bic = cdtr_agt.findtext('pacs:BICFI', namespaces=ns)
                if not cdtr_bic:
                    # Try clearing system member ID if BIC not found
                    cdtr_clr = cdtr_agt.find('.//pacs:ClrSysMmbId/pacs:MmbId', ns)
                    cdtr_routing = cdtr_clr.text if cdtr_clr is not None else None
                else:
                    cdtr_routing = cdtr_bic

            
            transaction = {
                'document': xml_content[:65000] if len(xml_content) > 65000 else xml_content,
                'instruction_id': self._safe_find_text(tx_inf, './/pacs:PmtId/pacs:InstrId', ns),
                'end_to_end_id': self._safe_find_text(tx_inf, './/pacs:PmtId/pacs:EndToEndId', ns),
                'amount': float(self._safe_find_text(tx_inf, './/pacs:IntrBkSttlmAmt', ns) or 0),
                'currency': tx_inf.find('.//pacs:IntrBkSttlmAmt', ns).get('Ccy') if tx_inf.find('.//pacs:IntrBkSttlmAmt', ns) is not None else None,
                'debtor_name': self._safe_find_text(tx_inf, './/pacs:Dbtr/pacs:Nm', ns),
                'debtor_account': extract_account_number_dbtr(tx_inf),
                'debtor_routing_number': dbtr_routing,
                'debtor_agent_bic': dbtr_bic,
                'creditor_name': self._safe_find_text(tx_inf, './/pacs:Cdtr/pacs:Nm', ns),
                'creditor_account': extract_account_number_cdtr(tx_inf),
                'creditor_routing_number': cdtr_routing,
                'creditor_agent_bic': cdtr_bic
            }
            transactions.append(transaction)
                
        return {'credit_transfers': transactions}
    
    def _parse_camt054(self, root: ET.Element, ns: Dict[str, str]) -> Dict:
        """Parse CAMT.054 message"""
        notifications = []
        xml_content = ET.tostring(root, encoding='unicode', method='xml')
        try:
            for ntfctn in root.findall('.//camt:BkToCstmrDbtCdtNtfctn/camt:Ntfctn', ns):
                entries = []
                
                for entry in ntfctn.findall('.//camt:Ntry', ns):
                    try:
                        # Safely extract amount and currency
                        amt_elem = entry.find('.//camt:Amt', ns)
                        amount = float(amt_elem.text) if amt_elem is not None else 0.0
                        currency = amt_elem.get('Ccy') if amt_elem is not None else None

                        # Extract transaction details including party information
                        tx_dtls = entry.find('.//camt:NtryDtls/camt:TxDtls', ns)
                        
                        # Extract debtor information
                        debtor_name = ""
                        debtor_account_number = ""
                        if tx_dtls is not None:
                            # Debtor name from RltdPties
                            debtor = tx_dtls.find('.//camt:RltdPties/camt:Dbtr/camt:Nm', ns)
                            if debtor is not None and debtor.text:
                                debtor_name = debtor.text.strip()
                            
                            # Debtor account from RltdPties
                            debtor_acct = tx_dtls.find('.//camt:RltdPties/camt:DbtrAcct/camt:Id/camt:Othr', ns)
                            if debtor_acct is not None and debtor_acct.text:
                                debtor_account_number = debtor_acct.text.strip()
                        
                        # Extract creditor information
                        creditor_name = ""
                        creditor_account_number = ""
                        if tx_dtls is not None:
                            # Creditor name from RltdPties
                            creditor = tx_dtls.find('.//camt:RltdPties/camt:Cdtr/camt:Nm', ns)
                            if creditor is not None and creditor.text:
                                creditor_name = creditor.text.strip()
                            
                            # Creditor account from RltdPties
                            creditor_acct = tx_dtls.find('.//camt:RltdPties/camt:CdtrAcct/camt:Id/camt:Othr', ns)
                            if creditor_acct is not None and creditor_acct.text:
                                creditor_account_number = creditor_acct.text.strip()
                        
                        # Fall back to the notification account info if debtor account not found in transaction details
                        if not debtor_account_number:
                            acct_info = ntfctn.find('.//camt:Acct/camt:Id/camt:Othr', ns)
                            if acct_info is not None and acct_info.text:
                                debtor_account_number = acct_info.text.strip()

                        # Safely get other fields with default values
                        entry_info = {
                            'document': xml_content[:65000] if len(xml_content) > 65000 else xml_content,
                            'amount': amount,
                            'currency': currency,
                            'credit_debit': self._safe_find_text(entry, './/camt:CdtDbtInd', ns) or '',
                            'status': self._safe_find_text(entry, './/camt:Sts', ns) or '',
                            'booking_date': self._safe_find_text(entry, './/camt:BookgDt/camt:Dt', ns) or '',
                            'value_date': self._safe_find_text(entry, './/camt:ValDt/camt:Dt', ns) or '',
                            'debtor_name': debtor_name,
                            'debtor_account': debtor_account_number,
                            'creditor_name': creditor_name,
                            'creditor_account': creditor_account_number,
                            'purpose_code': self._safe_find_text(tx_dtls, './/camt:Purp/camt:Cd', ns) or '',
                            'remittance_info': self._safe_find_text(tx_dtls, './/camt:RmtInf/camt:Ustrd', ns) or ''
                        }
                        entries.append(entry_info)
                    except Exception as e:
                        logger.warning(f"Error processing entry in CAMT.054: {str(e)}")
                        continue

                
                notification = {
                    'account': self._parse_account_info(ntfctn.find('.//camt:Acct', ns), 'camt', ns),
                    'entries': entries
                }
                notifications.append(notification)
        except Exception as e:
            logger.error(f"Error parsing CAMT.054 message: {str(e)}")
            # Return minimal valid structure instead of raising
            return {'notifications': [{'account': {}, 'entries': []}]}
            
        return {'notifications': notifications}
    
    
    def extract_account_number_dbtr(pacs_008_message, use_etree=True):
        """
        Extract account number from PACS.008 message
        
        Supports multiple extraction methods:
        1. Othr (Other) tag extraction
        2. IBAN extraction
        3. Regex fallback
        """
        import xml.etree.ElementTree as ET
        import re

        def _safe_find_text(element, xpath, namespaces=None):
            """
            Safely find and extract text from an XML element
            """
            if namespaces is None:
                namespaces = {
                    'pacs': 'urn:iso:std:iso:20022:tech:xsd:pacs.008.001.08'
                }
            
            try:
                found_element = element.find(xpath, namespaces)
                return found_element.text.strip() if found_element is not None else None
            except Exception:
                return None

        # Normalize input to handle both XML string and ElementTree
        if isinstance(pacs_008_message, str):
            try:
                root = ET.fromstring(pacs_008_message)
            except ET.ParseError:
                print("Invalid XML format")
                return None
        else:
            root = pacs_008_message

        # Namespaces dictionary
        ns = {
            'pacs': 'urn:iso:std:iso:20022:tech:xsd:pacs.008.001.08'
        }

        # 1. Try extracting from Othr tag
        othr_paths = [
            './/pacs:DbtrAcct/pacs:Id/pacs:Othr'
        ]
        
        for path in othr_paths:
            othr_element = root.find(path, ns)
            if othr_element is not None:
                account_number = othr_element.text
                if account_number:
                    return account_number.strip()

        # 2. Try extracting IBAN
        iban_paths = [
            './/pacs:DbtrAcct/pacs:Id/pacs:IBAN'
        ]
        
        for path in iban_paths:
            iban = _safe_find_text(root, path, ns)
            if iban:
                return iban

        # 3. Regex fallback method
        try:
            # Regex patterns to match account numbers in different formats
            account_patterns = [
                r'<pacs:Othr>([^<]+)</pacs:Othr>',
                r'<pacs:IBAN>([^<]+)</pacs:IBAN>'
            ]
            
            for pattern in account_patterns:
                match = re.search(pattern, ET.tostring(root).decode(), re.DOTALL)
                if match:
                    return match.group(1).strip()
        
        except Exception as e:
            print(f"Regex extraction failed: {e}")
        
        return None
    
    def extract_account_number_cdtr(pacs_008_message, use_etree=True):
        """
        Extract account number from PACS.008 message
        
        Supports multiple extraction methods:
        1. Othr (Other) tag extraction
        2. IBAN extraction
        3. Regex fallback
        """
        import xml.etree.ElementTree as ET
        import re

        def _safe_find_text(element, xpath, namespaces=None):
            """
            Safely find and extract text from an XML element
            """
            if namespaces is None:
                namespaces = {
                    'pacs': 'urn:iso:std:iso:20022:tech:xsd:pacs.008.001.08'
                }
            
            try:
                found_element = element.find(xpath, namespaces)
                return found_element.text.strip() if found_element is not None else None
            except Exception:
                return None

        # Normalize input to handle both XML string and ElementTree
        if isinstance(pacs_008_message, str):
            try:
                root = ET.fromstring(pacs_008_message)
            except ET.ParseError:
                print("Invalid XML format")
                return None
        else:
            root = pacs_008_message

        # Namespaces dictionary
        ns = {
            'pacs': 'urn:iso:std:iso:20022:tech:xsd:pacs.008.001.08'
        }

        # 1. Try extracting from Othr tag
        othr_paths = [
            './/pacs:CdtrAcct/pacs:Id/pacs:Othr'
        ]
        
        for path in othr_paths:
            othr_element = root.find(path, ns)
            if othr_element is not None:
                account_number = othr_element.text
                if account_number:
                    return account_number.strip()

        # 2. Try extracting IBAN
        iban_paths = [
            './/pacs:CdtrAcct/pacs:Id/pacs:IBAN'
        ]
        
        for path in iban_paths:
            iban = _safe_find_text(root, path, ns)
            if iban:
                return iban

        # 3. Regex fallback method
        try:
            # Regex patterns to match account numbers in different formats
            account_patterns = [
                r'<pacs:Othr>([^<]+)</pacs:Othr>',
                r'<pacs:IBAN>([^<]+)</pacs:IBAN>'
            ]
            
            for pattern in account_patterns:
                match = re.search(pattern, ET.tostring(root).decode(), re.DOTALL)
                if match:
                    return match.group(1).strip()
        
        except Exception as e:
            print(f"Regex extraction failed: {e}")
        
        return None


    def _safe_find_text(self, element: Optional[ET.Element], xpath: str, ns: Dict[str, str]) -> str:
        """Safely extract text from XML element with improved error handling"""
        try:
            if element is None:
                return ''
            found = element.find(xpath, ns)
            return self.safe_strip(found.text if found is not None else '')
        except Exception as e:
            logger.debug(f"Error in _safe_find_text: {str(e)}")
            return ''
    
    def _parse_party_info(self, party_element: Optional[ET.Element], ns_prefix: str, ns: Dict[str, str]) -> Dict[str, Any]:
        """Parse party information"""
        
        
        """Parse party information with comprehensive error handling"""
        result = {
            'name': '',
            'country': 'UNKNOWN',
            'address_lines': []
        }
        
        if party_element is None:
            return result
            
        try:
            name = self._safe_find_text(party_element, f'{ns_prefix}:Nm', ns)
            country = self._safe_find_text(party_element, f'{ns_prefix}:PstlAdr/{ns_prefix}:Ctry', ns)
            
            result['name'] = name
            result['country'] = country if country else 'UNKNOWN'
            
            # Safely get address lines
            address_lines = []
            for adr_line in party_element.findall(f'.//{ns_prefix}:AdrLine', ns):
                if adr_line is not None and adr_line.text:
                    address_lines.append(self.safe_strip(adr_line.text))
            result['address_lines'] = address_lines
            
        except Exception as e:
            logger.warning(f"Error parsing party info: {str(e)}")
            
        return result
    
    def _parse_account_info(self, account_element: Optional[ET.Element], ns_prefix: str, ns: Dict[str, str]) -> Dict[str, str]:
        """Parse account information"""
        
        result = {
            'iban': '',
            'other_id': ''
        }
        
        if account_element is None:
            return result
            
        try:
             # Try IBAN first
            iban = self._safe_find_text(account_element, f'./{ns_prefix}:Id/{ns_prefix}:IBAN', ns)
            
            # If IBAN is not found, look for Othr
            if not iban:
                other_id = self._safe_find_text(account_element, f'.//{ns_prefix}:Id/{ns_prefix}:Othr/{ns_prefix}:Id', ns)
                result['other_id'] = other_id
            else:
                result['iban'] = iban
            
            # If both IBAN and Othr are not found, try a more flexible search
            if not result['iban'] and not result['other_id']:
                # Try finding Othr directly
                othr_element = account_element.find(f'.//{ns_prefix}:Othr', ns)
                if othr_element is not None:
                    result['other_id'] = othr_element.text
            
        except Exception as e:
            logger.warning(f"Error parsing account info: {str(e)}")
            
        return result
        
    
    def _update_transaction_pain001(self, transaction: Dict, pmt_info: Dict, tx: Dict, high_risk_countries: Set[str]) -> None:
        """Update transaction with PAIN.001 specific information"""
        

        try:
            # Safe amount handling
            amount = 0.0
            currency = 'USD'
            
            if tx is not None:
                if isinstance(tx.get('amount'), dict):
                    amount = float(self.safe_get(tx['amount'], 'amount', 0))
                    currency = self.safe_get(tx['amount'], 'currency', 'USD')
                elif isinstance(tx.get('amount'), (int, float)):
                    amount = float(tx['amount'])
            
            transaction['amount'] = amount
            transaction['currency'] = currency

            # Safe debtor information
            debtor = self.safe_get(pmt_info, 'debtor', {})
            transaction['debtor_name'] = self.safe_get(debtor, 'name', '')
            transaction['debtor_country'] = self.safe_get(debtor, 'country', '')
            
            # Safe creditor information
            creditor = self.safe_get(tx, 'creditor', {})
            transaction['creditor_name'] = self.safe_get(creditor, 'name', '')
            transaction['creditor_country'] = self.safe_get(creditor, 'country', '')

            # Update accounts
            debtor_account = self.safe_get(pmt_info, 'debtor_account', {})
            creditor_account = self.safe_get(tx, 'creditor_account', {})
            
            transaction['debtor_account'] = self.safe_get(debtor_account, 'iban', '') or self.safe_get(debtor_account, 'other_id', '')
            transaction['creditor_account'] = self.safe_get(creditor_account, 'iban', '') or self.safe_get(creditor_account, 'other_id', '')

            # Update additional fields
            transaction['purpose_code'] = self.safe_get(tx, 'purpose', '')
            transaction['remittance_info'] = self.safe_get(tx, 'remittance_info', '')

            # Set flags
            transaction['is_cross_border'] = (
                transaction['debtor_country'] != transaction['creditor_country']
            )

            high_risk_countries = {'AF', 'KP', 'IR', 'MM', 'SS', 'SY', 'VE', 'YE'}
            transaction['involves_high_risk_country'] = any(
                country in high_risk_countries
                for country in [transaction['debtor_country'], transaction['creditor_country']]
            )

        except Exception as e:
            logger.error(f"Error updating PAIN.001 transaction: {str(e)}")

    def _update_transaction_pacs008(self, transaction: Dict, tx: Dict, high_risk_countries: Set[str]) -> None:
        """Update transaction with PACS.008 specific information"""
        
        try:
            # Safe amount handling
            transaction['amount'] = float(self.safe_get(tx, 'amount', 0))
            transaction['currency'] = self.safe_get(tx, 'currency', 'USD')

            # Safe party information
            transaction['debtor_name'] = self.safe_get(tx, 'debtor_name', '')
            transaction['debtor_account'] = self.safe_get(tx, 'debtor_account', '')
            transaction['creditor_name'] = self.safe_get(tx, 'creditor_name', '')
            transaction['creditor_account'] = self.safe_get(tx, 'creditor_account', '')

            # Extract countries safely
            debtor_country = self._extract_country_from_id(transaction['debtor_account'])
            creditor_country = self._extract_country_from_id(transaction['creditor_account'])
            
            transaction['debtor_country'] = debtor_country or 'UNKNOWN'
            transaction['creditor_country'] = creditor_country or 'UNKNOWN'

            # Set flags
            transaction['is_cross_border'] = (
                transaction['debtor_country'] != transaction['creditor_country']
            )

            high_risk_countries = {'AF', 'KP', 'IR', 'MM', 'SS', 'SY', 'VE', 'YE'}
            transaction['involves_high_risk_country'] = any(
                country in high_risk_countries
                for country in [transaction['debtor_country'], transaction['creditor_country']]
            )

        except Exception as e:
            logger.error(f"Error updating PACS.008 transaction: {str(e)}")

    def _update_transaction_camt054(self, transaction: Dict, notification: Dict, entry: Dict,high_risk_countries: Set[str]) -> None:
        """Update transaction with CAMT.054 specific information"""

        try:
            # Safe amount handling
            transaction['amount'] = float(self.safe_get(entry, 'amount', 0))
            transaction['currency'] = self.safe_get(entry, 'currency', 'USD')
            # Party information
            transaction['debtor_name'] = self.safe_get(entry, 'debtor_name', '')

            # For debtor account, prioritize the entry info but maintain existing key
            debtor_account = self.safe_get(entry, 'debtor_account', '')
            if debtor_account:
                transaction['debtor_account'] = debtor_account
            elif notification.get('account'):
                transaction['debtor_account'] = self.safe_get(notification['account'], 'iban', '')
            
            # Creditor information
            transaction['creditor_name'] = self.safe_get(entry, 'creditor_name', '')
            transaction['creditor_account'] = self.safe_get(entry, 'creditor_account', '')

            # Extract country information
            debtor_country = self._extract_country_from_id(transaction['debtor_account'])
            transaction['debtor_country'] = debtor_country or 'UNKNOWN'
            
            creditor_country = self._extract_country_from_id(transaction['creditor_account'])
            transaction['creditor_country'] = creditor_country or 'UNKNOWN'

            # Safe status handling
            transaction['credit_debit_indicator'] = self.safe_get(entry, 'credit_debit', '')
            transaction['status'] = self.safe_get(entry, 'status', '')

            # Safe date handling
            booking_date = self.safe_get(entry, 'booking_date', '')
            value_date = self.safe_get(entry, 'value_date', '')
            
            transaction['booking_date'] = self._parse_datetime(booking_date)
            transaction['value_date'] = self._parse_datetime(value_date)

            # Additional payment information
            transaction['purpose_code'] = self.safe_get(entry, 'purpose_code', '')
            transaction['remittance_info'] = self.safe_get(entry, 'remittance_info', '')

            # Set flags
            transaction['is_cross_border'] = transaction['debtor_country'] != transaction['creditor_country']

            transaction['involves_high_risk_country'] = (
                transaction['debtor_country'] in high_risk_countries or
                transaction['creditor_country'] in high_risk_countries
            )

        except Exception as e:
            logger.error(f"Error updating CAMT.054 transaction: {str(e)}")


    def _extract_country_from_id(self, identifier: str) -> Optional[str]:
        """Extract country code from BIC or IBAN"""
        if not identifier:
            return None
            
        # Try IBAN first (first two characters)
        if len(identifier) >= 2 and identifier[:2].isalpha():
            return identifier[:2].upper()
            
        # Try BIC (characters 5-6)
        if len(identifier) >= 6:
            country_code = identifier[4:6]
            if country_code.isalpha():
                return country_code.upper()
                
        return None

class TransactionAnalyzer:
    """Analyzes transaction patterns and computes metrics"""
    
    def __init__(self):
        self.lookback_periods = {
            '24h': timedelta(hours=24),
            '7d': timedelta(days=7),
            '30d': timedelta(days=30)
        }


    def _is_round_amount(self, amount):
        """Check if amount is suspiciously round (common in scams)"""
        try:
            # Check if amount is an exact multiple of 100, 500, 1000, etc.
            return (amount % 100 == 0 or 
                    amount % 500 == 0 or 
                    amount % 1000 == 0 or 
                    amount % 5000 == 0)
                    
        except Exception as e:
            logger.error(f"Error in round amount check: {str(e)}")
            return False

    def _calculate_normalized_entropy(self, counts):
        """Calculate normalized entropy (0-1 scale) of a distribution"""
        try:
            if counts.empty or counts.sum() == 0:
                return 0
                
            # Convert counts to probabilities
            probs = counts / counts.sum()
            
            # Calculate entropy
            entropy = -sum(p * np.log2(p) for p in probs if p > 0)
            
            # Normalize by maximum possible entropy
            max_entropy = np.log2(len(counts))
            if max_entropy > 0:
                return entropy / max_entropy
            return 0
            
        except Exception as e:
            logger.error(f"Error calculating entropy: {str(e)}")
            return 0

    def _calculate_beneficiary_risk_score(self, creditor_account, creditor_name):
        """Calculate risk score for beneficiary based on various factors"""
        try:
            base_score = 0
            
            # Check if recipient exists in known scam database
            if hasattr(self, 'scam_recipient_db') and self.scam_recipient_db:
                recipient_risk = self.scam_recipient_db.query_risk_score(creditor_account, creditor_name)
                if recipient_risk > 0:
                    return recipient_risk
                    
            # Add other beneficiary risk calculations here
            return base_score
            
        except Exception as e:
            logger.error(f"Error calculating beneficiary risk score: {str(e)}")
            return 0

    def compute_metrics(self, transaction: Dict, history_df: pd.DataFrame) -> TransactionMetrics:
        """Compute transaction metrics based on historical data with improved error handling"""
        try:
            # Safely extract channel with explicit None check
            payment_channel = None
            if transaction is not None and isinstance(transaction, dict):
                payment_channel = transaction.get('channel')
            
            # Default to 'UNKNOWN' if channel is None or empty
            if payment_channel is None or payment_channel == '':
                payment_channel = 'UNKNOWN'

            # Get transaction amount and convert to float to avoid decimal.Decimal comparison issues
            transaction_amount = 0.0
            if transaction is not None and 'amount' in transaction:
                try:
                    # Handle potential decimal.Decimal value
                    transaction_amount = float(transaction['amount'])
                except (ValueError, TypeError):
                    transaction_amount = 0.0

            # Create default metrics
            default_metrics = TransactionMetrics(
                velocity_24h=0,
                amount_24h=0.0,
                unique_recipients_24h=0,
                velocity_7d=0,
                amount_7d=0.0,
                unique_recipients_7d=0,
                avg_amount_30d=transaction_amount,
                std_amount_30d=0.0,
                new_recipient=True,
                cross_border=transaction.get('is_cross_border', False) if transaction is not None else False,
                high_risk_country=transaction.get('involves_high_risk_country', False) if transaction is not None else False,
                account_age_days=0,
                activity_days_before_transaction=0,
                days_since_first_transaction=0,
                days_since_last_transaction=0,
                weekday=datetime.now().weekday(),
                hour_of_day=datetime.now().hour,
                is_business_hours=(9 <= datetime.now().hour <= 17) and (datetime.now().weekday() < 5),
                rounded_amount_ratio=0.0,
                diverse_sources_count=0,
                funnel_pattern_ratio=0.0,
                off_hours_transaction_ratio=0.0,
                rapid_withdrawal_flag=False,
                account_emptying_percentage=0.0,
                velocity_ratio=1.0,
                amount_zscore=0.0,
                is_round_amount=self._is_round_amount(transaction_amount) if transaction is not None else False,
                app_fraud_risk=0.0,
                unusual_location_flag=False,
                geo_velocity_impossible_flag=False,
                distance_from_common_locations=0.0,
                unusual_access_time_flag=transaction.get('hour_of_day', 12) < 6 or transaction.get('hour_of_day', 12) > 22 if transaction is not None else False
            )

            # Return defaults if no history or transaction is None
            if transaction is None or history_df is None or history_df.empty:
                return default_metrics
            
            try:
                # Ensure history_df is a DataFrame
                if not isinstance(history_df, pd.DataFrame):
                    logger.warning("history_df is not a DataFrame, returning default metrics")
                    return default_metrics
                    
                # Make a copy to avoid modifying the original DataFrame
                history_copy = history_df.copy()
                
                # Ensure amount column is float type to avoid decimal.Decimal issues
                if 'amount' in history_copy.columns:
                    try:
                        history_copy['amount'] = history_copy['amount'].astype(float)
                    except Exception as amount_error:
                        logger.error(f"Error converting amounts to float: {amount_error}")
                        # Try element-wise conversion
                        history_copy['amount'] = history_copy['amount'].apply(
                            lambda x: float(x) if x is not None else 0.0
                        )
                    
                # Normalize dates in the DataFrame
                normalized_df = normalize_dataframe_dates(history_copy)
                if normalized_df.empty:
                    return default_metrics
                    
                # Add required columns if missing
                if 'amount' not in normalized_df.columns:
                    normalized_df['amount'] = 0.0
                    
                if 'creditor_account' not in normalized_df.columns:
                    normalized_df['creditor_account'] = ''
                
                # Get channel history only if the channel column exists
                channel_history = normalized_df
                if 'channel' in normalized_df.columns and payment_channel != 'UNKNOWN':
                    channel_history = normalized_df[normalized_df['channel'] == payment_channel]
                    # If filtering by channel results in empty DataFrame, fall back to all history
                    if channel_history.empty:
                        channel_history = normalized_df
                
                # Normalize transaction date
                transaction_date = normalize_datetime(transaction.get('creation_date'))
                
                # Filter for different time periods
                recent_24h = filter_by_timespan(channel_history, transaction_date, 24)
                recent_7d = filter_by_timespan(channel_history, transaction_date, 24 * 7)
                recent_30d = filter_by_timespan(channel_history, transaction_date, 24 * 30)
                recent_365d = filter_by_timespan(channel_history, transaction_date, 24 * 365)
                
                # Get recipient account safely
                creditor_account = transaction.get('creditor_account', '')
                
                # Calculate account temporal metrics
                if not normalized_df.empty and 'creation_date' in normalized_df.columns:
                    # Sort by creation_date
                    normalized_df = normalized_df.sort_values('creation_date')
                    
                    # Calculate days since first transaction
                    if len(normalized_df) > 0:
                        first_transaction_date = normalized_df['creation_date'].min()
                        days_since_first_transaction = (transaction_date - first_transaction_date).days
                        default_metrics.days_since_first_transaction = days_since_first_transaction
                        
                        # Calculate account age if account creation date is available
                        if 'account_creation_date' in transaction and transaction['account_creation_date']:
                            account_creation_date = transaction['account_creation_date']
                            account_age_days = (transaction_date - account_creation_date).days
                            default_metrics.account_age_days = account_age_days
                        
                        # Activity days before transaction
                        unique_activity_days = normalized_df['creation_date'].dt.date.nunique()
                        default_metrics.activity_days_before_transaction = unique_activity_days
                        
                        # Days since last transaction
                        last_transaction_date = normalized_df['creation_date'].max()
                        days_since_last_transaction = (transaction_date - last_transaction_date).days
                        default_metrics.days_since_last_transaction = days_since_last_transaction
                
                # Handle potential empty DataFrames
                amount_24h = 0.0
                if not recent_24h.empty and 'amount' in recent_24h.columns:
                    try:
                        amount_24h = float(recent_24h['amount'].sum())
                    except (ValueError, TypeError):
                        amount_24h = 0.0
                        
                amount_7d = 0.0
                if not recent_7d.empty and 'amount' in recent_7d.columns:
                    try:
                        amount_7d = float(recent_7d['amount'].sum())
                    except (ValueError, TypeError):
                        amount_7d = 0.0
                    
                avg_amount_30d = transaction_amount
                if not recent_30d.empty and 'amount' in recent_30d.columns and len(recent_30d) > 0:
                    try:
                        avg_amount_30d = float(recent_30d['amount'].mean())
                    except (ValueError, TypeError):
                        avg_amount_30d = transaction_amount
                    
                std_amount_30d = 0.0
                if not recent_30d.empty and 'amount' in recent_30d.columns and len(recent_30d) > 1:
                    try:
                        std_amount_30d = float(recent_30d['amount'].std())
                    except (ValueError, TypeError):
                        std_amount_30d = 0.0
                    
                # Calculate unique recipients safely
                unique_recipients_24h = 0
                if not recent_24h.empty and 'creditor_account' in recent_24h.columns:
                    unique_recipients_24h = len(recent_24h['creditor_account'].unique())
                    
                unique_recipients_7d = 0
                if not recent_7d.empty and 'creditor_account' in recent_7d.columns:
                    unique_recipients_7d = len(recent_7d['creditor_account'].unique())
                    
                # Calculate Z-score for amount
                amount_zscore = 0.0
                if std_amount_30d > 0:
                    amount_zscore = abs((transaction_amount - avg_amount_30d) / std_amount_30d)
                    
                # Calculate transaction pattern metrics
                if not normalized_df.empty and 'amount' in normalized_df.columns:
                    # Round amount ratio
                    total_transactions = len(normalized_df)
                    if total_transactions > 0:
                        rounded_amounts = normalized_df[
                            (normalized_df['amount'] % 100 == 0) | 
                            (normalized_df['amount'] % 500 == 0) | 
                            (normalized_df['amount'] % 1000 == 0)
                        ]
                        rounded_amount_ratio = len(rounded_amounts) / total_transactions
                        default_metrics.rounded_amount_ratio = rounded_amount_ratio
                        
                        # Check if current transaction amount is round
                        default_metrics.is_round_amount = self._is_round_amount(transaction_amount)
                    
                    # Funnel pattern ratio (incoming vs outgoing)
                    if 'direction' in normalized_df.columns:
                        incoming = normalized_df[normalized_df['direction'] == 'INCOMING']
                        outgoing = normalized_df[normalized_df['direction'] == 'OUTGOING']
                        
                        if len(outgoing) > 0:
                            funnel_ratio = len(incoming) / len(outgoing)
                            default_metrics.funnel_pattern_ratio = funnel_ratio
                    
                    # Off-hours transaction ratio
                    if 'creation_date' in normalized_df.columns:
                        off_hours = normalized_df[
                            (normalized_df['creation_date'].dt.hour < 9) | 
                            (normalized_df['creation_date'].dt.hour > 17) |
                            (normalized_df['creation_date'].dt.dayofweek >= 5)  # Weekend
                        ]
                        off_hours_ratio = len(off_hours) / total_transactions if total_transactions > 0 else 0
                        default_metrics.off_hours_transaction_ratio = off_hours_ratio
                
                # Calculate source diversity
                if not normalized_df.empty and 'debtor_account' in normalized_df.columns:
                    recent_90d = filter_by_timespan(normalized_df, transaction_date, 24 * 90)
                    unique_sources = recent_90d['debtor_account'].nunique()
                    default_metrics.diverse_sources_count = unique_sources
                
                # Calculate rapid withdrawal flag and account emptying percentage
                if not normalized_df.empty and 'amount' in normalized_df.columns and 'account_balance_before' in normalized_df.columns:
                    withdrawals = normalized_df[normalized_df['amount'] < 0]
                    
                    if not withdrawals.empty:
                        # Check for rapid withdrawals after deposits
                        deposits = normalized_df[normalized_df['amount'] > 0].sort_values('creation_date')
                        
                        if not deposits.empty:
                            for _, deposit in deposits.iterrows():
                                # Find withdrawals within 24 hours of deposit
                                deposit_time = deposit['creation_date']
                                next_day = deposit_time + pd.Timedelta(days=1)
                                rapid_withdrawals = withdrawals[
                                    (withdrawals['creation_date'] > deposit_time) & 
                                    (withdrawals['creation_date'] <= next_day)
                                ]
                                
                                if len(rapid_withdrawals) > 0:
                                    default_metrics.rapid_withdrawal_flag = True
                                    break
                        
                        # Calculate account emptying percentage
                        if 'account_balance_before' in transaction and transaction['account_balance_before'] > 0:
                            emptying_pct = abs(transaction_amount) / transaction['account_balance_before'] * 100
                            default_metrics.account_emptying_percentage = emptying_pct
                
                # Calculate velocity ratio
                if not normalized_df.empty and 'creation_date' in normalized_df.columns:
                    # Current velocity (daily)
                    current_velocity = len(recent_24h)
                    
                    # Historical average velocity
                    days_span = max((transaction_date - normalized_df['creation_date'].min()).days, 1)
                    avg_velocity = len(normalized_df) / days_span
                    
                    if avg_velocity > 0:
                        velocity_ratio = current_velocity / avg_velocity
                        default_metrics.velocity_ratio = velocity_ratio
                
                # Calculate seasonal metrics using 365-day lookback
                if not recent_365d.empty and 'creation_date' in recent_365d.columns and len(recent_365d) > 90:
                    # Quarterly comparison
                    current_quarter = pd.Timestamp(transaction_date).quarter
                    current_year = pd.Timestamp(transaction_date).year
                    
                    # Get transactions from current quarter
                    current_quarter_data = recent_365d[
                        (recent_365d['creation_date'].dt.quarter == current_quarter) & 
                        (recent_365d['creation_date'].dt.year == current_year)
                    ]
                    
                    # Get transactions from previous quarter
                    if current_quarter > 1:
                        prev_quarter = current_quarter - 1
                        prev_quarter_year = current_year
                    else:
                        prev_quarter = 4
                        prev_quarter_year = current_year - 1
                    
                    prev_quarter_data = recent_365d[
                        (recent_365d['creation_date'].dt.quarter == prev_quarter) & 
                        (recent_365d['creation_date'].dt.year == prev_quarter_year)
                    ]
                    
                    if not current_quarter_data.empty and not prev_quarter_data.empty:
                        # Compare average transaction amounts
                        curr_avg = current_quarter_data['amount'].mean()
                        prev_avg = prev_quarter_data['amount'].mean()
                        
                        if prev_avg > 0:
                            quarterly_change = (curr_avg - prev_avg) / prev_avg
                            default_metrics.quarterly_activity_comparison = quarterly_change
                    
                    # Year-over-year comparison
                    prev_year_data = recent_365d[
                        recent_365d['creation_date'].dt.year == current_year - 1
                    ]
                    
                    if not prev_year_data.empty:
                        curr_year_data = recent_365d[
                            recent_365d['creation_date'].dt.year == current_year
                        ]
                        
                        if not curr_year_data.empty:
                            # Compare transaction patterns
                            curr_year_avg = curr_year_data['amount'].mean()
                            prev_year_avg = prev_year_data['amount'].mean()
                            
                            if prev_year_avg > 0:
                                yoy_change = (curr_year_avg - prev_year_avg) / prev_year_avg
                                default_metrics.year_over_year_behavior_change = yoy_change
                    
                    # Calculate total unique recipients in 365 days
                    if 'creditor_account' in recent_365d.columns:
                        default_metrics.total_unique_recipients_365d = recent_365d['creditor_account'].nunique()
                    
                    # Calculate transaction frequency metrics
                    if len(recent_365d) > 30:
                        # Group by month and count
                        recent_365d['month'] = recent_365d['creation_date'].dt.to_period('M')
                        monthly_counts = recent_365d.groupby('month').size()
                        
                        # Monthly transaction count average
                        default_metrics.avg_monthly_transaction_count = monthly_counts.mean()
                        
                        # Transaction frequency standard deviation
                        default_metrics.transaction_frequency_stddev = monthly_counts.std()
                        
                        # Monthly transaction volume
                        monthly_volumes = recent_365d.groupby('month')['amount'].sum()
                        default_metrics.avg_monthly_transaction_volume = monthly_volumes.mean()
                        
                        # Calculate amount volatility
                        default_metrics.amount_volatility_365d = recent_365d['amount'].std() / recent_365d['amount'].mean() if recent_365d['amount'].mean() > 0 else 0
                        
                        # Detect cyclical amount pattern breaks
                        if len(monthly_volumes) >= 12:
                            # Simple check: compare month-over-month changes
                            mom_changes = monthly_volumes.pct_change()
                            if (mom_changes.abs() > 0.5).any():  # Detect 50% changes
                                default_metrics.cyclical_amount_pattern_break = True
                        
                        # Calculate payment growth rate (simple linear regression)
                        if len(monthly_volumes) > 1:
                            try:
                                x = np.arange(len(monthly_volumes))
                                y = monthly_volumes.values
                                slope, _ = np.polyfit(x, y, 1)
                                # Calculate payment growth rate as percentage
                                if monthly_volumes.iloc[0] != 0:
                                    growth_rate = slope / monthly_volumes.iloc[0] * 100
                                    default_metrics.payment_growth_rate = growth_rate
                            except Exception as e:
                                logger.error(f"Error calculating payment growth rate: {str(e)}")
                
                # Calculate behavioral consistency score
                if 'amount' in recent_365d.columns and len(recent_365d) > 30:
                    # Group transactions by different time periods
                    recent_365d['day_of_week'] = recent_365d['creation_date'].dt.dayofweek
                    recent_365d['hour_of_day'] = recent_365d['creation_date'].dt.hour
                    
                    # Check consistency in day of week
                    dow_counts = recent_365d['day_of_week'].value_counts()
                    dow_entropy = self._calculate_normalized_entropy(dow_counts)
                    
                    # Check consistency in hour of day
                    hour_counts = recent_365d['hour_of_day'].value_counts()
                    hour_entropy = self._calculate_normalized_entropy(hour_counts)
                    
                    # Check consistency in amount
                    amount_cv = recent_365d['amount'].std() / recent_365d['amount'].mean() if recent_365d['amount'].mean() > 0 else 1
                    
                    # Combine into overall consistency score (higher is more consistent)
                    consistency_score = 100 * (1 - (dow_entropy * 0.3 + hour_entropy * 0.3 + min(amount_cv, 1) * 0.4))
                    default_metrics.behavioral_consistency_score = consistency_score
                    
                    # Determine typical transaction hour and weekday
                    top_hours = hour_counts.nlargest(3).index.tolist()
                    top_weekdays = dow_counts.nlargest(3).index.tolist()
                    
                    # Check if current transaction matches typical patterns
                    current_hour = transaction_date.hour
                    current_weekday = transaction_date.weekday()
                    
                    default_metrics.typical_transaction_hour_flag = current_hour in top_hours
                    default_metrics.typical_transaction_weekday_flag = current_weekday in top_weekdays
                    
                    # Count pattern breaks in 365 days
                    pattern_breaks = 0
                    if not recent_365d.empty and 'amount' in recent_365d.columns:
                        # Use rolling window to detect pattern breaks
                        if len(recent_365d) > 10:
                            # Sort by date
                            recent_365d = recent_365d.sort_values('creation_date')
                            
                            # Calculate rolling stats
                            rolling_mean = recent_365d['amount'].rolling(window=10).mean()
                            rolling_std = recent_365d['amount'].rolling(window=10).std()
                            
                            # Count significant deviations
                            for i in range(10, len(recent_365d)):
                                if rolling_std.iloc[i] > 0:
                                    z_score = abs((recent_365d['amount'].iloc[i] - rolling_mean.iloc[i]) / rolling_std.iloc[i])
                                    if z_score > 3.0:  # Significant deviation
                                        pattern_breaks += 1
                    
                    default_metrics.pattern_break_frequency_365d = pattern_breaks
                        
                    # Calculate transaction predictability score
                    predictability_components = []
                    
                    # Time predictability (higher entropy = less predictable)
                    time_predictability = 100 * (1 - hour_entropy)
                    predictability_components.append(time_predictability)
                    
                    # Day predictability
                    day_predictability = 100 * (1 - dow_entropy)
                    predictability_components.append(day_predictability)
                    
                    # Amount predictability (based on coefficient of variation)
                    amount_predictability = 100 * (1 - min(amount_cv, 1))
                    predictability_components.append(amount_predictability)
                    
                    # Recipient predictability
                    if 'creditor_account' in recent_365d.columns:
                        recipient_counts = recent_365d['creditor_account'].value_counts()
                        recipient_entropy = self._calculate_normalized_entropy(recipient_counts)
                        recipient_predictability = 100 * (1 - recipient_entropy)
                        predictability_components.append(recipient_predictability)
                    
                    # Overall predictability score
                    default_metrics.transaction_predictability_score = sum(predictability_components) / len(predictability_components)
            
                # Calculate beneficiary-specific metrics
                creditor_account = transaction.get('creditor_account', '')
                
                if creditor_account and not normalized_df.empty and 'creditor_account' in normalized_df.columns:
                    # Filter transactions to this recipient
                    recipient_history = normalized_df[normalized_df['creditor_account'] == creditor_account]
                    
                    if not recipient_history.empty:
                        # Days since first payment to recipient
                        first_payment_date = recipient_history['creation_date'].min()
                        days_since_first = (transaction_date - first_payment_date).days
                        default_metrics.days_since_first_payment_to_recipient = days_since_first
                        
                        # Check for dormant reactivation (no transactions to recipient in last 180 days)
                        recent_to_recipient = recipient_history[
                            recipient_history['creation_date'] >= (transaction_date - pd.Timedelta(days=180))
                        ]
                        
                        if len(recent_to_recipient) <= 1 and len(recipient_history) > 1:
                            default_metrics.dormant_recipient_reactivation = True
                        
                        # Previous transfers count
                        default_metrics.source_target_previous_transfers = len(recipient_history)
                        
                        # Calculate recipient inactivity period
                        if len(recipient_history) > 1:
                            # Sort by date
                            recipient_history = recipient_history.sort_values('creation_date')
                            
                            # Calculate time differences between transactions
                            recipient_history['next_date'] = recipient_history['creation_date'].shift(-1)
                            recipient_history['days_between'] = (recipient_history['next_date'] - recipient_history['creation_date']).dt.days
                            
                            # Get maximum gap
                            max_gap = recipient_history['days_between'].max()
                            if not pd.isna(max_gap):
                                default_metrics.recipient_inactivity_period = int(max_gap)
                    
                    # Check for pattern change in recipient behavior
                    if not recipient_history.empty:
                        # Compare current transaction to typical behavior with this recipient
                        if 'amount' in recipient_history.columns and len(recipient_history) > 0:
                            avg_amount_to_recipient = recipient_history['amount'].mean()
                            std_amount_to_recipient = recipient_history['amount'].std() if len(recipient_history) > 1 else 0
                            
                            if std_amount_to_recipient > 0:
                                recipient_z_score = abs((transaction_amount - avg_amount_to_recipient) / std_amount_to_recipient)
                                
                                # Flag if significantly different
                                if recipient_z_score > 2.0:
                                    default_metrics.recipient_pattern_change_flag = True
                        
                        # Calculate new recipient large amount flag
                        if default_metrics.new_recipient and transaction_amount > avg_amount_30d * 2:
                            default_metrics.new_recipient_large_amount = True
                    
                    # Calculate recipient frequency distribution
                    if 'creditor_account' in normalized_df.columns:
                        recipient_counts = normalized_df['creditor_account'].value_counts()
                        total_transactions = len(normalized_df)
                        
                        if total_transactions > 0:
                            # Calculate concentration ratio (% of transactions to top recipients)
                            top_recipients = recipient_counts.nlargest(3)
                            top_concentration = top_recipients.sum() / total_transactions
                            default_metrics.recipient_concentration_ratio = top_concentration
                            
                            # Store frequency distribution
                            frequency_dist = {}
                            for recipient, count in recipient_counts.items():
                                frequency_dist[recipient] = count / total_transactions
                            
                            default_metrics.recipient_frequency_distribution = frequency_dist
                
                # Create a compact transaction history for the current recipient
                if creditor_account and not normalized_df.empty and 'creditor_account' in normalized_df.columns:
                    recipient_history = normalized_df[normalized_df['creditor_account'] == creditor_account]
                    
                    if not recipient_history.empty and len(recipient_history) < 100:  # Limit size
                        compact_history = []
                        for _, row in recipient_history.iterrows():
                            compact_history.append({
                                'date': row['creation_date'].isoformat() if hasattr(row['creation_date'], 'isoformat') else str(row['creation_date']),
                                'amount': float(row['amount']) if 'amount' in row else 0.0,
                            })
                        
                        default_metrics.recipient_transaction_history_365d = compact_history
                
                # Check for recipient is new
                new_recipient = True
                if creditor_account and not recent_30d.empty and 'creditor_account' in recent_30d.columns:
                    new_recipient = creditor_account not in recent_30d['creditor_account'].values
                    default_metrics.new_recipient = new_recipient
                    default_metrics.new_beneficiary_flag = new_recipient
                
                # Calculate behavioral risk scores
                # Transaction pattern break
                if default_metrics.quarterly_activity_comparison != 0:
                    pattern_break_score = min(abs(default_metrics.quarterly_activity_comparison) * 50, 100)
                    default_metrics.transaction_pattern_break_risk_score = pattern_break_score
                
                # Amount anomaly
                if amount_zscore > 0:
                    amount_anomaly_score = min(amount_zscore * 25, 100)
                    default_metrics.amount_anomaly_risk_score = amount_anomaly_score
                    default_metrics.amount_pattern_risk_score = amount_anomaly_score
                
                # New recipient
                if new_recipient:
                    default_metrics.new_recipient_risk_score = 65.0
                    default_metrics.new_beneficiary_risk_score = 65.0
                
                # Round amount
                if default_metrics.is_round_amount:
                    default_metrics.round_amount_risk_score = 60.0
                
                # Unusual timing
                if not default_metrics.is_business_hours:
                    default_metrics.unusual_timing_risk_score = 50.0
                    default_metrics.after_hours_risk_score = 55.0
                    default_metrics.unusual_time_risk_score = 50.0
                
                # Velocity spike
                if default_metrics.velocity_ratio > 3.0:
                    velocity_risk = min(default_metrics.velocity_ratio * 20, 100)
                    default_metrics.velocity_risk_score = velocity_risk
                
                # Multiple beneficiaries risk
                if unique_recipients_24h > 3:
                    multiple_beneficiaries_risk = min(unique_recipients_24h * 20, 100)
                    default_metrics.multiple_beneficiaries_risk_score = multiple_beneficiaries_risk
                
                # Beneficiary risk score calculation
                if default_metrics.new_recipient_large_amount or default_metrics.dormant_recipient_reactivation:
                    default_metrics.beneficiary_risk_score = 70.0
                elif default_metrics.recipient_pattern_change_flag:
                    default_metrics.beneficiary_risk_score = 60.0
                
                # Calculate triggered indicators count
                triggered_indicators = 0
                if default_metrics.amount_anomaly_risk_score > 50:
                    triggered_indicators += 1
                if default_metrics.new_beneficiary_risk_score > 50:
                    triggered_indicators += 1
                if default_metrics.velocity_risk_score > 50:
                    triggered_indicators += 1
                if default_metrics.unusual_timing_risk_score > 50:
                    triggered_indicators += 1
                if default_metrics.multiple_beneficiaries_risk_score > 50:
                    triggered_indicators += 1
                
                default_metrics.triggered_indicators_count = triggered_indicators
                default_metrics.multi_indicator_present = triggered_indicators > 1
                
                # Calculate aggregated risk scores
                risk_scores = [
                    default_metrics.transaction_pattern_break_risk_score,
                    default_metrics.amount_anomaly_risk_score,
                    default_metrics.new_recipient_risk_score,
                    default_metrics.round_amount_risk_score,
                    default_metrics.unusual_timing_risk_score,
                    default_metrics.velocity_risk_score,
                    default_metrics.multiple_beneficiaries_risk_score
                ]
                
                # Filter out zeros
                active_scores = [score for score in risk_scores if score > 0]
                
                if active_scores:
                    # Calculate aggregated risk score
                    aggregated_risk = sum(active_scores) / len(active_scores)
                    default_metrics.aggregated_risk_score = aggregated_risk
                    
                    # Apply multi-indicator bonus
                    if triggered_indicators > 1:
                        combined_risk = aggregated_risk + (triggered_indicators - 1) * 10
                        combined_risk = min(combined_risk, 100)
                    else:
                        combined_risk = aggregated_risk
                    
                    default_metrics.combined_risk = combined_risk
                    
                    # Apply channel-specific adjustment
                    channel_multiplier = 1.0
                    if payment_channel == 'ZELLE':
                        channel_multiplier = 1.2
                    elif payment_channel == 'WIRE':
                        channel_multiplier = 1.1
                    elif payment_channel == 'ACH':
                        channel_multiplier = 0.9
                    
                    channel_adjusted = combined_risk * channel_multiplier
                    channel_adjusted = min(channel_adjusted, 100)
                    default_metrics.channel_adjusted_score = channel_adjusted
                    
                    # Set final risk score
                    if default_metrics.beneficiary_risk_score > 0:
                        recipient_risk = default_metrics.beneficiary_risk_score
                        default_metrics.recipient_risk = recipient_risk
                        
                        # Use maximum for final score
                        final_risk = max(channel_adjusted, recipient_risk)
                    else:
                        final_risk = channel_adjusted
                    
                    # Set account takeover flags based on final risk score
                    if final_risk > 80:
                        default_metrics.is_suspected_takeover = True
                        if triggered_indicators >= 3:
                            default_metrics.is_account_takeover = True
                    
                    # Set APP fraud flag based on similar logic
                    if final_risk > 75 and (default_metrics.new_recipient or default_metrics.round_amount_risk_score > 0):
                        default_metrics.is_suspected_app_fraud = True
                        default_metrics.app_fraud_risk = final_risk
                    
                    # Set confidence level
                    if final_risk >= 80:
                        default_metrics.confidence_level = "HIGH"
                    elif final_risk >= 60:
                        default_metrics.confidence_level = "MEDIUM"
                    else:
                        default_metrics.confidence_level = "LOW"
                
                # Set standard metrics
                default_metrics.velocity_24h = len(recent_24h)
                default_metrics.amount_24h = float(amount_24h)
                default_metrics.unique_recipients_24h = unique_recipients_24h
                default_metrics.velocity_7d = len(recent_7d)
                default_metrics.amount_7d = float(amount_7d)
                default_metrics.unique_recipients_7d = unique_recipients_7d
                default_metrics.avg_amount_30d = float(avg_amount_30d)
                default_metrics.std_amount_30d = float(std_amount_30d)
                
                # Also set aliases for backward compatibility
                default_metrics.amount_30d_avg = float(avg_amount_30d)
                default_metrics.amount_30d_std = float(std_amount_30d)
                default_metrics.avg_velocity_30d = len(recent_30d) / 30 if len(recent_30d) > 0 else 0
                
                return default_metrics
                
            except Exception as e:
                logger.error(f"Error computing metrics details: {str(e)}")
                return default_metrics
                    
        except Exception as e:
            logger.error(f"Error in compute_metrics: {str(e)}")
            return self._create_default_metrics()


    def _create_default_metrics(self, is_first_transaction: bool = False, 
                          transaction_amount: float = 0.0,
                          is_cross_border: bool = False,
                          is_high_risk_country: bool = False) -> TransactionMetrics:
        """Create default metrics for error cases"""
        return TransactionMetrics(
            # Existing parameters
            velocity_24h=0,
            amount_24h=0.0,
            unique_recipients_24h=0,
            velocity_7d=0,
            amount_7d=0.0,
            unique_recipients_7d=0,
            avg_amount_30d=transaction_amount if is_first_transaction else 0.0,
            std_amount_30d=0.0,
            new_recipient=True,
            cross_border=is_cross_border,
            high_risk_country=is_high_risk_country,
            
            # New parameters - all with safe defaults
            account_age_days=0,
            activity_days_before_transaction=0,
            days_since_first_transaction=0,
            days_since_last_transaction=0,
            weekday=datetime.now().weekday(),
            hour_of_day=datetime.now().hour,
            is_business_hours=(9 <= datetime.now().hour <= 17) and (datetime.now().weekday() < 5),
            typical_transaction_hour_flag=False,
            typical_transaction_weekday_flag=False,
            rounded_amount_ratio=0.0,
            diverse_sources_count=0,
            funnel_pattern_ratio=0.0,
            off_hours_transaction_ratio=0.0,
            rapid_withdrawal_flag=False,
            account_emptying_percentage=0.0,
            velocity_ratio=1.0,
            amount_zscore=0.0,
            seasonal_adjustment_factor=1.0,
            seasonal_pattern_deviation=0.0,
            quarterly_activity_comparison=0.0,
            year_over_year_behavior_change=0.0,
            new_beneficiary_flag=True,
            beneficiary_risk_score=0.0,
            recipient_pattern_change_flag=False,
            days_since_first_payment_to_recipient=999,
            dormant_recipient_reactivation=False,
            recipient_concentration_ratio=0.0,
            total_unique_recipients_365d=0,
            recipient_network_density=0.0,
            new_network_connections=0,
            source_target_previous_transfers=0,
            transaction_pattern_break_risk_score=0.0,
            amount_anomaly_risk_score=0.0,
            new_recipient_risk_score=0.0,
            round_amount_risk_score=0.0,
            unusual_timing_risk_score=0.0,
            after_hours_risk_score=0.0,
            unusual_location_risk_score=0.0,
            velocity_risk_score=0.0,
            is_round_amount=self._is_round_amount(transaction_amount) if transaction_amount else False,
            app_fraud_risk=0.0,
            known_scam_recipient_flag=False,
            threshold_days_since_last_recipient=999,
            unusual_location_flag=False,
            geo_velocity_impossible_flag=False,
            distance_from_common_locations=0.0,
            # mule_layering_pattern=False,
            # mule_rapid_disbursement=False,
            # mule_new_account_high_volume=False,
            # mule_geographic_diversity=0,
            new_recipient_large_amount=False,
            high_risk_corridor_flag=False,
            collect_transfer_pattern_flag=False,
            source_account_risk_score=0.0,
            target_account_risk_score=0.0,
            annual_amount_trend=0.0,
            payment_growth_rate=0.0,
            cyclical_amount_pattern_break=False,
            amount_volatility_365d=0.0,
            account_milestone_deviation=0.0,
            major_behavior_shift_timestamps=[],
            account_maturity_risk_score=0.0,
            lifecycle_stage_appropriate_behavior=True,
            behavioral_consistency_score=0.0,
            pattern_break_frequency_365d=0,
            transaction_predictability_score=0.0,
            rolling_risk_trend_365d=0.0,
            behavioral_change_velocity=0.0,
            risk_volatility_365d=0.0,
            historical_fraud_attempts=0,
            aggregated_risk_score=0.0,
            combined_risk=0.0,
            channel_adjusted_score=0.0,
            recipient_risk=0.0,
            behavioral_score=0.0,
            confidence_level="LOW",
            multiple_beneficiaries_risk_score=0.0,
            unusual_time_risk_score=0.0,
            triggered_indicators_count=0,
            is_suspected_app_fraud=False,
            is_suspected_takeover=False,
            is_account_takeover=False,
            multi_indicator_present=False,
            avg_velocity_30d=0.0,
            amount_30d_avg=transaction_amount if is_first_transaction else 0.0,
            amount_30d_std=0.0,
            periodic_payment_disruption=False,
            avg_monthly_transaction_count=0.0,
            avg_monthly_transaction_volume=0.0,
            transaction_frequency_stddev=0.0,
            recipient_inactivity_period=0,
            recipient_frequency_distribution={},
            recipient_transaction_history_365d=[],
            login_frequency_30d=0.0,
            device_change_recency=999,
            credential_age_days=999,
            failed_login_attempts_24h=0,
            password_reset_recency=999,
            mfa_bypass_attempts=0,
            mfa_status=False,
            is_known_device=True,
            device_reputation_score=0.0,
            browser_fingerprint_match=True,
            session_duration=0,
            session_anomaly_score=0.0,
            ip_address_change_flag=False,
            vpn_proxy_flag=False,
            unusual_access_time_flag=False
        )

class TransactionFeatureExtractor:
    """Extracts and preprocesses features for ML models"""
    
    def __init__(self, config_reader=None,channel_detector=None):
        """Initialize the feature extractor with config reader"""
        self.config_reader = config_reader
        # print(f"DEBUG: TransactionFeatureExtractor.__init__ - self.config_reader is now {type(self.config_reader)}")
        
        # Add a fallback for missing config_reader
        if self.config_reader is None:
            # print("WARNING: config_reader is None in TransactionFeatureExtractor.__init__, trying to create a default")
            try:
                from configreader import ConfigReader
                self.config_reader = ConfigReader('./config.properties')
                # print(f"DEBUG: Created default ConfigReader: {type(self.config_reader)}")
            except Exception as e:
                print(f"DEBUG: Could not create default ConfigReader: {e}")
        
        self.channel_detector = channel_detector
        self.default_config = {
            'zelle.threshold.high_amount': '1000',
            'wire.threshold.high_amount': '50000',
            'swift.threshold.high_amount': '100000',
            'fednow.threshold.high_amount': '25000',
            'ach.threshold.high_amount': '10000',
            'risk.threshold.high': '70',
            'zelle.threshold.daily_limit': '3000',
            'zelle.threshold.max_recipients': '5',
            'zelle.threshold.velocity_24h': '6'
            # Add other commonly used properties
        }

        self.scaler = StandardScaler()
        self.categorical_features = ['channel', 'currency', 'debtor_country', 'creditor_country']
        
        # Updated numerical features list
        self.numerical_features = [
            # Transaction basics
            'amount', 
            
            # Velocity and volume metrics
            'velocity_24h', 'amount_24h', 'unique_recipients_24h',
            'velocity_7d', 'amount_7d', 'unique_recipients_7d',
            'avg_amount_30d', 'std_amount_30d', 'amount_30d_avg', 'amount_30d_std',
            'avg_velocity_30d',
            
            # Account-level temporal features
            'account_age_days', 'activity_days_before_transaction',
            'days_since_first_transaction', 'days_since_last_transaction',
            
            # Transaction pattern features
            'amount_zscore', 'rounded_amount_ratio', 'diverse_sources_count',
            'funnel_pattern_ratio', 'off_hours_transaction_ratio',
            'account_emptying_percentage', 'velocity_ratio',
            
            # Temporal behavior
            'weekday', 'hour_of_day',
            
            # Seasonal patterns
            'seasonal_adjustment_factor', 'seasonal_pattern_deviation',
            'quarterly_activity_comparison', 'year_over_year_behavior_change',
            
            # Beneficiary features
            'beneficiary_risk_score', 'days_since_first_payment_to_recipient',
            'recipient_concentration_ratio', 'recipient_inactivity_period',
            
            # Network metrics
            'total_unique_recipients_365d', 'recipient_network_density',
            'new_network_connections', 'source_target_previous_transfers',
            
            # Risk scores
            'transaction_pattern_break_risk_score', 'amount_anomaly_risk_score',
            'new_recipient_risk_score', 'new_beneficiary_risk_score',
            'round_amount_risk_score', 'unusual_timing_risk_score',
            'after_hours_risk_score', 'unusual_location_risk_score',
            'velocity_risk_score', 'multiple_beneficiaries_risk_score',
            'amount_pattern_risk_score', 'unusual_time_risk_score',
            
            # Location features
            'distance_from_common_locations',
            
            # Authentication and device metrics
            'login_frequency_30d', 'device_change_recency',
            'credential_age_days', 'failed_login_attempts_24h',
            'password_reset_recency', 'mfa_bypass_attempts',
            'device_reputation_score', 'session_duration', 'session_anomaly_score',
            
            # Mule detection features
            'mule_geographic_diversity', 'source_account_risk_score', 'target_account_risk_score',
            
            # Long-term pattern analysis
            'annual_amount_trend', 'payment_growth_rate',
            'amount_volatility_365d', 'account_milestone_deviation',
            'account_maturity_risk_score',
            
            # Behavioral consistency metrics
            'behavioral_consistency_score', 'pattern_break_frequency_365d',
            'transaction_predictability_score',
            
            # Risk trending
            'rolling_risk_trend_365d', 'behavioral_change_velocity',
            'risk_volatility_365d', 'historical_fraud_attempts',
            
            # Aggregated risk metrics
            'aggregated_risk_score', 'combined_risk',
            'channel_adjusted_score', 'recipient_risk',
            'behavioral_score', 'app_fraud_risk',
            
            # Flag indicators
            'triggered_indicators_count',
            
            # Transaction count and volume metrics
            'avg_monthly_transaction_count', 'avg_monthly_transaction_volume',
            'transaction_frequency_stddev',
            
            # Boolean flags (converted to numeric)
            'is_new_recipient', 'new_recipient', 'new_beneficiary_flag',
            'is_cross_border', 'is_high_risk_country', 'involves_high_risk_country', 
            'is_business_hours', 'is_high_risk_hour',
            'typical_transaction_hour_flag', 'typical_transaction_weekday_flag',
            'unusual_access_time_flag', 'periodic_payment_disruption',
            'rapid_withdrawal_flag', 'is_round_amount',
            'recipient_pattern_change_flag', 'dormant_recipient_reactivation',
            'new_recipient_large_amount', 'unusual_location_flag',
            'geo_velocity_impossible_flag', 'vpn_proxy_flag',
            'ip_address_change_flag', 'mfa_status', 'is_known_device',
            'browser_fingerprint_match',
            # 'mule_layering_pattern',
            # 'mule_rapid_disbursement', 'mule_new_account_high_volume',
            'high_risk_corridor_flag', 'collect_transfer_pattern_flag',
            'cyclical_amount_pattern_break', 'lifecycle_stage_appropriate_behavior',
            'known_scam_recipient_flag', 'is_suspected_app_fraud',
            'is_suspected_takeover', 'is_account_takeover',
            'multi_indicator_present',
            
            # Existing derived features
            'tx_count_1h', 'tx_count_24h', 'route_count_1h', 'route_count_24h'
            ]
        
        self.is_fitted = False
        self.feature_names = []  # Store feature names after encoding
        
        # Fixed categories for each feature to ensure consistency
        self.fixed_categories = {
            'channel': ['WIRE', 'ACH', 'SWIFT', 'ZELLE', 'FEDNOW', 'UNKNOWN'],
            'currency': ['USD', 'EUR', 'GBP', 'UNKNOWN'],
            'debtor_country': ['US', 'GB', 'DE', 'FR', 'UNKNOWN'],
            'creditor_country': ['US', 'GB', 'DE', 'FR', 'UNKNOWN']
        }

        def get_property(self, key, default):
            """Safe property access with multiple fallbacks"""
            try:
                # Try getting from config_reader first
                if self.config_reader is not None:
                    return self.config_reader.get_property(key, default)
                
                # If no config_reader, try default config dictionary
                if key in self.default_config:
                    return self.default_config[key]
                    
                # Final fallback to provided default
                return default
            except Exception as e:
                logger.error(f"Error accessing property {key}: {str(e)}")
                return default
        
        self.scaler = StandardScaler()
        self.categorical_features = ['channel', 'currency', 'debtor_country', 'creditor_country']
        self.numerical_features = [
            'amount', 'velocity_24h', 'amount_24h', 'unique_recipients_24h',
            'velocity_7d', 'amount_7d', 'unique_recipients_7d',
            'avg_amount_30d', 'std_amount_30d'
        ]
        self.is_fitted = False
        self.feature_names = []  # Store feature names after encoding
        
        # Fixed categories for each feature to ensure consistency
        self.fixed_categories = {
            'channel': ['WIRE', 'ACH', 'SWIFT', 'ZELLE', 'FEDNOW', 'UNKNOWN'],
            'currency': ['USD', 'EUR', 'GBP', 'UNKNOWN'],
            'debtor_country': ['US', 'GB', 'DE', 'FR', 'UNKNOWN'],
            'creditor_country': ['US', 'GB', 'DE', 'FR', 'UNKNOWN']
        }

    def _handle_nan_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle NaN values in DataFrame"""
        # Fill NaN values in numerical features with 0
        for col in self.numerical_features:
            df[col] = df[col].fillna(0)
        
        # Fill NaN values in categorical features with 'UNKNOWN'
        for col in self.categorical_features:
            df[col] = df[col].fillna('UNKNOWN')
        
        return df
    
    def _extract_account_metrics(self, account_history_df, transaction):
        """
        Extract advanced account-level metrics for mule detection
        
        Args:
            account_history_df: DataFrame with account transaction history
            transaction: Current transaction dictionary
        
        Returns:
            Dictionary with extracted metrics
        """
        account_metrics = {
            'unique_sources_7d': 0,
            'unique_destinations_7d': 0,
            'off_hours_txn_ratio': 0.0,
            'rounded_amount_ratio': 0.0,
            'min_time_to_withdrawal_minutes': 999999,
            'max_withdrawal_pct': 0.0,
            'geographic_diversity': 0,
            'account_age_days': 999,
            'tx_count_7d': 0,
            'is_suspected_mule': False,
            'recipient_risk_score': 0,
            'rapid_collect_transfer': False,
            'is_new_recipient': True
        }
        
        # Early return for empty history
        if account_history_df is None or account_history_df.empty:
            return account_metrics
        
        try:
            # Get current transaction details
            txn_date = transaction.get('creation_date')
            if txn_date is None:
                txn_date = datetime.now()
            
            # Calculate account age if creation date is available
            if 'account_creation_date' in transaction and transaction['account_creation_date']:
                account_age_days = (txn_date - transaction['account_creation_date']).days
                account_metrics['account_age_days'] = account_age_days
            
            # Filter for recent history (7 days)
            recent_df = account_history_df[
                account_history_df['creation_date'] >= txn_date - timedelta(days=7)
            ]
            
            account_metrics['tx_count_7d'] = len(recent_df)
            
            # Source and destination diversity
            if 'debtor_account' in recent_df.columns:
                # Count unique source accounts (deposits)
                deposits_df = recent_df[recent_df['amount'] > 0]
                account_metrics['unique_sources_7d'] = deposits_df['debtor_account'].nunique()
            
            if 'creditor_account' in recent_df.columns:
                # Count unique destination accounts (withdrawals)
                withdrawals_df = recent_df[recent_df['amount'] < 0]
                account_metrics['unique_destinations_7d'] = withdrawals_df['creditor_account'].nunique()
                
                # Check if current recipient is new
                if 'creditor_account' in transaction:
                    account_metrics['is_new_recipient'] = transaction['creditor_account'] not in withdrawals_df['creditor_account'].values
            
            # Off-hours transaction ratio
            if 'creation_date' in recent_df.columns:
                total_txns = len(recent_df)
                if total_txns > 0:
                    # Count off-hours transactions (before 5am or after 10pm)
                    off_hours_txns = len(recent_df[
                        (recent_df['creation_date'].dt.hour < 5) | 
                        (recent_df['creation_date'].dt.hour >= 22)
                    ])
                    account_metrics['off_hours_txn_ratio'] = off_hours_txns / total_txns
            
            # Rounded amount ratio
            if 'amount' in recent_df.columns:
                deposits_df = recent_df[recent_df['amount'] > 0]
                total_deposits = len(deposits_df)
                if total_deposits > 0:
                    # Count deposits with rounded amounts
                    rounded_deposits = len(deposits_df[
                        (deposits_df['amount'] % 100 == 0) | 
                        (deposits_df['amount'] % 50 == 0) | 
                        (deposits_df['amount'] % 25 == 0)
                    ])
                    account_metrics['rounded_amount_ratio'] = rounded_deposits / total_deposits
            
            # Time to withdrawal analysis
            if 'amount' in recent_df.columns and 'creation_date' in recent_df.columns:
                deposits_df = recent_df[recent_df['amount'] > 0].sort_values('creation_date')
                withdrawals_df = recent_df[recent_df['amount'] < 0].sort_values('creation_date')
                
                if not deposits_df.empty and not withdrawals_df.empty:
                    # Find the minimum time between a deposit and withdrawal
                    min_time = None
                    
                    for _, deposit in deposits_df.iterrows():
                        # Find withdrawals that occurred after this deposit
                        subsequent_withdrawals = withdrawals_df[
                            withdrawals_df['creation_date'] > deposit['creation_date']
                        ]
                        
                        if not subsequent_withdrawals.empty:
                            # Calculate time to first withdrawal
                            time_diff = (subsequent_withdrawals.iloc[0]['creation_date'] - 
                                        deposit['creation_date']).total_seconds() / 60
                            
                            if min_time is None or time_diff < min_time:
                                min_time = time_diff
                    
                    if min_time is not None:
                        account_metrics['min_time_to_withdrawal_minutes'] = min_time
            
            # Maximum withdrawal percentage
            if ('amount' in recent_df.columns and 
                'creation_date' in recent_df.columns and 
                'account_balance_before' in recent_df.columns):
                
                withdrawals_df = recent_df[recent_df['amount'] < 0]
                
                if not withdrawals_df.empty:
                    # Calculate withdrawal percentage for each transaction
                    withdrawal_pcts = []
                    
                    for _, withdrawal in withdrawals_df.iterrows():
                        balance_before = withdrawal['account_balance_before']
                        if balance_before > 0:
                            withdrawal_pct = abs(withdrawal['amount']) / balance_before * 100
                            withdrawal_pcts.append(withdrawal_pct)
                    
                    if withdrawal_pcts:
                        account_metrics['max_withdrawal_pct'] = max(withdrawal_pcts)
            
            # Geographic diversity
            if 'ip_location' in recent_df.columns:
                account_metrics['geographic_diversity'] = recent_df['ip_location'].nunique()
            
            # Rapid collect and transfer pattern
            if 'amount' in recent_df.columns and 'creation_date' in recent_df.columns:
                # Get last 3 days of activity
                very_recent_df = recent_df[
                    recent_df['creation_date'] >= txn_date - timedelta(days=3)
                ]
                
                if not very_recent_df.empty:
                    deposits = very_recent_df[very_recent_df['amount'] > 0]
                    withdrawals = very_recent_df[very_recent_df['amount'] < 0]
                    
                    if not deposits.empty and not withdrawals.empty:
                        total_deposits = deposits['amount'].sum()
                        total_withdrawals = abs(withdrawals['amount'].sum())
                        
                        # Pattern: Multiple deposits followed by a large withdrawal
                        if (len(deposits) >= 3 and 
                            len(withdrawals) > 0 and 
                            total_withdrawals >= 0.7 * total_deposits):
                            
                            account_metrics['rapid_collect_transfer'] = True
            
            return account_metrics
        
        except Exception as e:
            logger.error(f"Error extracting account metrics: {str(e)}")
            return account_metrics

    def get_config_value(self, key, default_value):
        """Safe method to get config values with fallback"""
        # print(f"DEBUG: get_config_value called for key='{key}', default='{default_value}'")
        # print(f"DEBUG: self.config_reader is {type(self.config_reader)}")
        try:
            if hasattr(self, 'config_reader') and self.config_reader is not None:
                # print(f"DEBUG: Using config_reader.get_property for key='{key}'")
                value = self.config_reader.get_property(key, default_value)
                # print(f"DEBUG: Retrieved value '{value}' for key='{key}'")
                return self.config_reader.get_property(key, default_value)
            elif key in self.default_thresholds:
                # print(f"DEBUG: Using default_thresholds for key='{key}'")
                return self.default_thresholds[key]
            else:
                # print(f"DEBUG: Using provided default_value='{default_value}' for key='{key}'")
                return default_value
        except Exception as e:
            # Handle error and return default
            return default_value
    
    def get_channel_detector_property(self, channel, property_name, default_value):
        """Safely get a property from channel_detector"""
        try:
            if self.channel_detector is None:
                self.logger.warning(f"Channel detector is None, using default for {property_name}: {default_value}")
                return default_value
            # Replace with appropriate method to access channel detector properties
            return default_value  # Modify this to properly access channel detector properties
        except Exception as e:
            self.logger.error(f"Error getting channel property {property_name}: {str(e)}")
            return default_value
        
    def fit(self, transactions: List[Dict]):
        """Fit the feature extractor to historical data"""

        # Save reference to config_reader since it might get lost
        saved_config_reader = self.config_reader if hasattr(self, 'config_reader') else None
    
        # logger.info("Starting feature extractor fitting...")
        try:
            # Convert transactions to DataFrame
            df = pd.DataFrame(transactions)
            # logger.info(f"Created DataFrame with shape: {df.shape}")
            
            # Create features with defaults
            features_df = pd.DataFrame([{
                'amount': float(tx.get('amount', 0)),
                'channel': str(tx.get('channel', 'UNKNOWN')),
                'currency': str(tx.get('currency', 'USD')),
                'debtor_country': str(tx.get('debtor_country', 'UNKNOWN')),
                'creditor_country': str(tx.get('creditor_country', 'UNKNOWN')),
                'velocity_24h': 0.0,
                'amount_24h': 0.0,
                'unique_recipients_24h': 0.0,
                'velocity_7d': 0.0,
                'amount_7d': 0.0,
                'unique_recipients_7d': 0.0,
                'avg_amount_30d': 0.0,
                'std_amount_30d': 0.0
            } for tx in transactions])
            
            # Encode categorical features
            encoded_df = self._encode_categorical(features_df)
            
            # Fit scaler on numerical features
            numerical_data = encoded_df[self.numerical_features].values
            self.scaler.fit(numerical_data)
            # logger.info("Fitted scaler successfully")
            
            self.is_fitted = True
            # logger.info(f"Feature extractor fitted, total features: {len(self.feature_names)}")
            return self
            
        except Exception as e:
            logger.error(f"Error in fit: {str(e)}")
            raise
            
    def __getstate__(self):
        """Control what gets pickled - exclude non-serializable objects"""
        state = self.__dict__.copy()
        # Don't pickle these attributes
        if 'config_reader' in state:
            del state['config_reader']
        if 'channel_detector' in state:
            del state['channel_detector']
        return state

    def __setstate__(self, state):
        """Control unpickling - initialize with defaults for missing attributes"""
        self.__dict__.update(state)
        # These will be re-set after loading
        self.config_reader = None
        self.channel_detector = None
    
    def extract_features(self, transaction: Dict, metrics: TransactionMetrics) -> np.ndarray:
        """Extract features from transaction and metrics"""
        # print(f"DEBUG: extract_features called for transaction with channel='{transaction.get('channel', 'UNKNOWN')}'")
        # print(f"DEBUG: self.config_reader is {type(self.config_reader)}")

        # Add defense for missing config_reader
        if not hasattr(self, 'config_reader') or self.config_reader is None:
            default_thresholds = {
                'ZELLE': {'high_amount': 1000.0},
                'ACH': {'high_amount': 10000.0},
                'WIRE': {'high_amount': 50000.0},
                'SWIFT': {'high_amount': 100000.0},
                'FEDNOW': {'high_amount': 25000.0}
            }

        if not hasattr(self, 'channel_detector') or self.channel_detector is None:
            # Create a minimal substitute that won't cause errors
            class EmptyChannelDetector:
                def __init__(self):
                    self.suspicious_odfi_list = []
                    self.suspicious_rdfi_list = []
            
            self.channel_detector = EmptyChannelDetector()
        

        # Convert boolean and string values to numeric
        def to_numeric(value):
            if isinstance(value, bool):
                return 1.0 if value else 0.0
            elif isinstance(value, (int, float)):
                return float(value)
            elif isinstance(value, str):
                if value.lower() == 'true':
                    return 1.0
                elif value.lower() == 'false':
                    return 0.0
                try:
                    return float(value)
                except ValueError:
                    return 0.0
            return 0.0
        
        if not self.is_fitted:
            raise ValueError("TransactionFeatureExtractor must be fitted before extracting features")
        
        try:
    
            # Get payment channel
            payment_channel = transaction.get('channel', 'UNKNOWN')

            def get_config_value(key, default_value):
                try:
                    if hasattr(self, 'config_reader') and self.config_reader is not None:
                        return self.config_reader.get_property(key, default_value)
                    elif hasattr(self, 'default_config') and key in self.default_config:
                        return self.default_config[key]
                    else:
                        return default_value
                except Exception as e:
                    logger.error(f"Error getting config value for {key}: {str(e)}")
                    return default_value
            
            
            # Use get_config_value throughout the method instead of direct access
            high_amount_threshold = float(get_config_value(
                f'{payment_channel.lower()}.threshold.high_amount', 
                10000.0
            ))

            # Get country code and current time
            country_code = transaction.get('debtor_country', 'UNKNOWN')
            transaction_time = transaction.get('creation_date', datetime.now(timezone.utc))

            # Use safe config access
            if not hasattr(self, 'config_reader') or self.config_reader is None:
                # Use default values
                print("WARNING: config_reader is None, using default thresholds")
                # Set default thresholds
                thresholds = {
                    'ZELLE': 1000.0,
                    'ACH': 10000.0,
                    'WIRE': 50000.0,
                    'SWIFT': 100000.0,
                    'FEDNOW': 25000.0
                }
                high_amount_threshold = thresholds.get(payment_channel, 10000.0)
            else:
                # Use config reader to get thresholds
                high_amount_threshold = float(self.config_reader.get_property(
                    f'{payment_channel.lower()}.threshold.high_amount', 
                    10000.0
                ))

            # Initialize TimeZoneManager if needed
            if not hasattr(self, 'timezone_manager'):
                self.timezone_manager = TimeZoneManager()

             # Get channel-specific settings from config
            try:
                if payment_channel == 'ZELLE':
                    business_start = int(get_config_value('zelle.business_hours.start', '9'))
                    business_end = int(get_config_value('zelle.business_hours.end', '17'))
                    allow_weekend = get_config_value('zelle.allow_weekend', 'false').lower() == 'true'
                elif payment_channel == 'ACH':
                    business_start = int(get_config_value('ach.business_hours.start', '9'))
                    business_end = int(get_config_value('ach.business_hours.end', '17'))
                    allow_weekend = get_config_value('ach.allow_weekend', 'false').lower() == 'true'
                elif payment_channel == 'WIRE':
                    business_start = int(get_config_value('wire.business_hours.start', '9'))
                    business_end = int(get_config_value('wire.business_hours.end', '17'))
                    allow_weekend = get_config_value('wire.allow_weekend', 'false').lower() == 'true'
                elif payment_channel == 'SWIFT':
                    business_start = int(get_config_value('swift.business_hours.start', '9'))
                    business_end = int(get_config_value('swift.business_hours.end', '17'))
                    allow_weekend = get_config_value('swift.allow_weekend', 'false').lower() == 'true'
                elif payment_channel == 'FEDNOW':
                    business_start = int(get_config_value('fednow.business_hours.start', '9'))
                    business_end = int(get_config_value('fednow.business_hours.end', '17'))
                    allow_weekend = get_config_value('fednow.allow_weekend', 'false').lower() == 'true'
                else:
                    business_start = 9
                    business_end = 17
                    allow_weekend = False
            except Exception as e:
                logger.warning(f"Error getting channel-specific settings: {str(e)}")
                business_start = 9
                business_end = 17
                allow_weekend = False
            
            # Convert to local time for the debtor's country
            if country_code != 'UNKNOWN':
                # Get local time hour
                local_time = self.timezone_manager.convert_to_local_time(transaction_time, country_code)
                local_hour = local_time.hour
                is_weekend = local_time.weekday() >= 5  
                # Apply channel-specific business hours logic with weekend handling
                is_business_hours = (business_start <= local_hour < business_end) 
                if is_weekend:
                    is_business_hours = is_business_hours and allow_weekend
                # is_business_hours = (9 <= local_hour <= 17) and (local_time.weekday() < 5)
                is_high_risk_hour = not is_business_hours
            else:
                # Fallback to server time
                local_hour = transaction_time.hour
                is_weekend = datetime.now().weekday() >= 5  
                # is_business_hours = 9 <= local_hour <= 17
                # Apply channel-specific business hours logic with weekend handling
                is_business_hours = (business_start <= local_hour < business_end)
                if is_weekend:
                    is_business_hours = is_business_hours and allow_weekend
                is_high_risk_hour = not is_business_hours

            # Create features dictionary with explicit numeric conversion
            features = {
                'amount': to_numeric(transaction.get('amount', 0)),
                'channel': str(transaction.get('channel', 'UNKNOWN')),
                'currency': str(transaction.get('currency', 'USD')),
                'debtor_country': str(transaction.get('debtor_country', 'UNKNOWN')),
                'creditor_country': str(transaction.get('creditor_country', 'UNKNOWN')),
                'creditor_account': str(transaction.get('creditor_account', '')),
                'creditor_name': str(transaction.get('creditor_name', '')),
                
                # Velocity and volume metrics
                'velocity_24h': to_numeric(metrics.velocity_24h if metrics else 0),
                'amount_24h': to_numeric(metrics.amount_24h if metrics else 0),
                'unique_recipients_24h': to_numeric(metrics.unique_recipients_24h if metrics else 0),
                'velocity_7d': to_numeric(metrics.velocity_7d if metrics else 0),
                'amount_7d': to_numeric(metrics.amount_7d if metrics else 0),
                'unique_recipients_7d': to_numeric(metrics.unique_recipients_7d if metrics else 0),
                'avg_amount_30d': to_numeric(metrics.avg_amount_30d if metrics else 0),
                'std_amount_30d': to_numeric(metrics.std_amount_30d if metrics else 0),
                'amount_30d_avg': to_numeric(metrics.avg_amount_30d if metrics else 0),
                'amount_30d_std': to_numeric(metrics.std_amount_30d if metrics else 0),
                'avg_velocity_30d': to_numeric(metrics.avg_velocity_30d if metrics else 0),
                
                # Account-level temporal features
                'account_age_days': to_numeric(metrics.account_age_days if metrics else 0),
                'activity_days_before_transaction': to_numeric(metrics.activity_days_before_transaction if metrics else 0),
                'days_since_first_transaction': to_numeric(metrics.days_since_first_transaction if metrics else 0),
                'days_since_last_transaction': to_numeric(metrics.days_since_last_transaction if metrics else 0),
                
                # Transaction pattern features
                'is_new_recipient': to_numeric(metrics.new_recipient if metrics else True),
                'new_recipient': to_numeric(metrics.new_recipient if metrics else True),  # Alias
                'new_beneficiary_flag': to_numeric(metrics.new_beneficiary_flag if metrics else True),
                'is_cross_border': to_numeric(metrics.cross_border if metrics else False),
                'is_high_risk_country': to_numeric(metrics.high_risk_country if metrics else False),
                'involves_high_risk_country': to_numeric(metrics.high_risk_country if metrics else False),  # Alias
                'amount_zscore': to_numeric(metrics.amount_zscore if metrics else 0),
                'rounded_amount_ratio': to_numeric(metrics.rounded_amount_ratio if metrics else 0),
                'diverse_sources_count': to_numeric(metrics.diverse_sources_count if metrics else 0),
                'funnel_pattern_ratio': to_numeric(metrics.funnel_pattern_ratio if metrics else 0),
                'off_hours_transaction_ratio': to_numeric(metrics.off_hours_transaction_ratio if metrics else 0),
                'rapid_withdrawal_flag': to_numeric(metrics.rapid_withdrawal_flag if metrics else False),
                'account_emptying_percentage': to_numeric(metrics.account_emptying_percentage if metrics else 0),
                'velocity_ratio': to_numeric(metrics.velocity_ratio if metrics else 1.0),
                'is_round_amount': to_numeric(metrics.is_round_amount if metrics else False),
                
                # Temporal behavior
                'weekday': to_numeric(metrics.weekday if metrics else datetime.now().weekday()),
                'hour_of_day': to_numeric(metrics.hour_of_day if metrics else datetime.now().hour),
                'is_business_hours': to_numeric(metrics.is_business_hours if metrics else True),
                'is_high_risk_hour': to_numeric(is_high_risk_hour),
                'typical_transaction_hour_flag': to_numeric(metrics.typical_transaction_hour_flag if metrics else False),
                'typical_transaction_weekday_flag': to_numeric(metrics.typical_transaction_weekday_flag if metrics else False),
                'unusual_access_time_flag': to_numeric(metrics.unusual_access_time_flag if metrics else False),
                
                # Seasonal patterns
                'seasonal_adjustment_factor': to_numeric(metrics.seasonal_adjustment_factor if metrics else 1.0),
                'seasonal_pattern_deviation': to_numeric(metrics.seasonal_pattern_deviation if metrics else 0),
                'quarterly_activity_comparison': to_numeric(metrics.quarterly_activity_comparison if metrics else 0),
                'year_over_year_behavior_change': to_numeric(metrics.year_over_year_behavior_change if metrics else 0),
                'periodic_payment_disruption': to_numeric(metrics.periodic_payment_disruption if metrics else False),
                
                # Beneficiary features
                'beneficiary_risk_score': to_numeric(metrics.beneficiary_risk_score if metrics else 0),
                'recipient_pattern_change_flag': to_numeric(metrics.recipient_pattern_change_flag if metrics else False),
                'days_since_first_payment_to_recipient': to_numeric(metrics.days_since_first_payment_to_recipient if metrics else 999),
                'dormant_recipient_reactivation': to_numeric(metrics.dormant_recipient_reactivation if metrics else False),
                'recipient_concentration_ratio': to_numeric(metrics.recipient_concentration_ratio if metrics else 0),
                'new_recipient_large_amount': to_numeric(metrics.new_recipient_large_amount if metrics else False),
                'source_target_previous_transfers': to_numeric(metrics.source_target_previous_transfers if metrics else 0),
                'recipient_inactivity_period': to_numeric(metrics.recipient_inactivity_period if metrics else 0),
                
                # Network metrics
                'total_unique_recipients_365d': to_numeric(metrics.total_unique_recipients_365d if metrics else 0),
                'recipient_network_density': to_numeric(metrics.recipient_network_density if metrics else 0),
                'new_network_connections': to_numeric(metrics.new_network_connections if metrics else 0),
                
                # Risk scores
                'transaction_pattern_break_risk_score': to_numeric(metrics.transaction_pattern_break_risk_score if metrics else 0),
                'amount_anomaly_risk_score': to_numeric(metrics.amount_anomaly_risk_score if metrics else 0),
                'amount_pattern_risk_score': to_numeric(metrics.amount_pattern_risk_score if metrics else 0),
                'new_recipient_risk_score': to_numeric(metrics.new_recipient_risk_score if metrics else 0),
                'new_beneficiary_risk_score': to_numeric(metrics.new_beneficiary_risk_score if metrics else 0),
                'round_amount_risk_score': to_numeric(metrics.round_amount_risk_score if metrics else 0),
                'unusual_timing_risk_score': to_numeric(metrics.unusual_timing_risk_score if metrics else 0),
                'unusual_time_risk_score': to_numeric(metrics.unusual_time_risk_score if metrics else 0),
                'after_hours_risk_score': to_numeric(metrics.after_hours_risk_score if metrics else 0),
                'unusual_location_risk_score': to_numeric(metrics.unusual_location_risk_score if metrics else 0),
                'velocity_risk_score': to_numeric(metrics.velocity_risk_score if metrics else 0),
                'multiple_beneficiaries_risk_score': to_numeric(metrics.multiple_beneficiaries_risk_score if metrics else 0),
                
                # Location features
                'unusual_location_flag': to_numeric(metrics.unusual_location_flag if metrics else False),
                'geo_velocity_impossible_flag': to_numeric(metrics.geo_velocity_impossible_flag if metrics else False),
                'distance_from_common_locations': to_numeric(metrics.distance_from_common_locations if metrics else 0),
                'vpn_proxy_flag': to_numeric(metrics.vpn_proxy_flag if metrics else False),
                'ip_address_change_flag': to_numeric(metrics.ip_address_change_flag if metrics else False),
                
                # Authentication and device metrics
                'login_frequency_30d': to_numeric(metrics.login_frequency_30d if metrics else 0),
                'device_change_recency': to_numeric(min(metrics.device_change_recency if metrics else 999, 999)),
                'credential_age_days': to_numeric(min(metrics.credential_age_days if metrics else 999, 999)),
                'failed_login_attempts_24h': to_numeric(metrics.failed_login_attempts_24h if metrics else 0),
                'password_reset_recency': to_numeric(min(metrics.password_reset_recency if metrics else 999, 999)),
                'mfa_bypass_attempts': to_numeric(metrics.mfa_bypass_attempts if metrics else 0),
                'mfa_status': to_numeric(metrics.mfa_status if metrics else False),
                'is_known_device': to_numeric(metrics.is_known_device if metrics else True),
                'device_reputation_score': to_numeric(metrics.device_reputation_score if metrics else 0),
                'browser_fingerprint_match': to_numeric(metrics.browser_fingerprint_match if metrics else True),
                'session_duration': to_numeric(metrics.session_duration if metrics else 0),
                'session_anomaly_score': to_numeric(metrics.session_anomaly_score if metrics else 0),
                
                # Mule detection features
                # 'mule_layering_pattern': to_numeric(metrics.mule_layering_pattern if metrics else False),
                # 'mule_rapid_disbursement': to_numeric(metrics.mule_rapid_disbursement if metrics else False),
                # 'mule_new_account_high_volume': to_numeric(metrics.mule_new_account_high_volume if metrics else False),
                # 'mule_geographic_diversity': to_numeric(metrics.mule_geographic_diversity if metrics else 0),
                # 'high_risk_corridor_flag': to_numeric(metrics.high_risk_corridor_flag if metrics else False),
                # 'collect_transfer_pattern_flag': to_numeric(metrics.collect_transfer_pattern_flag if metrics else False),
                # 'source_account_risk_score': to_numeric(metrics.source_account_risk_score if metrics else 0),
                # 'target_account_risk_score': to_numeric(metrics.target_account_risk_score if metrics else 0),
                
                # Long-term pattern analysis
                'annual_amount_trend': to_numeric(metrics.annual_amount_trend if metrics else 0),
                'payment_growth_rate': to_numeric(metrics.payment_growth_rate if metrics else 0),
                'cyclical_amount_pattern_break': to_numeric(metrics.cyclical_amount_pattern_break if metrics else False),
                'amount_volatility_365d': to_numeric(metrics.amount_volatility_365d if metrics else 0),
                'account_milestone_deviation': to_numeric(metrics.account_milestone_deviation if metrics else 0),
                'account_maturity_risk_score': to_numeric(metrics.account_maturity_risk_score if metrics else 0),
                'lifecycle_stage_appropriate_behavior': to_numeric(metrics.lifecycle_stage_appropriate_behavior if metrics else True),
                
                # Behavioral consistency metrics
                'behavioral_consistency_score': to_numeric(metrics.behavioral_consistency_score if metrics else 0),
                'pattern_break_frequency_365d': to_numeric(metrics.pattern_break_frequency_365d if metrics else 0),
                'transaction_predictability_score': to_numeric(metrics.transaction_predictability_score if metrics else 0),
                
                # Risk trending
                'rolling_risk_trend_365d': to_numeric(metrics.rolling_risk_trend_365d if metrics else 0),
                'behavioral_change_velocity': to_numeric(metrics.behavioral_change_velocity if metrics else 0),
                'risk_volatility_365d': to_numeric(metrics.risk_volatility_365d if metrics else 0),
                'historical_fraud_attempts': to_numeric(metrics.historical_fraud_attempts if metrics else 0),
                
                # Aggregated risk metrics
                'aggregated_risk_score': to_numeric(metrics.aggregated_risk_score if metrics else 0),
                'combined_risk': to_numeric(metrics.combined_risk if metrics else 0),
                'channel_adjusted_score': to_numeric(metrics.channel_adjusted_score if metrics else 0),
                'recipient_risk': to_numeric(metrics.recipient_risk if metrics else 0),
                'behavioral_score': to_numeric(metrics.behavioral_score if metrics else 0),
                'app_fraud_risk': to_numeric(metrics.app_fraud_risk if metrics else 0),
                
                # Flag indicators
                'known_scam_recipient_flag': to_numeric(metrics.known_scam_recipient_flag if metrics else False),
                'triggered_indicators_count': to_numeric(metrics.triggered_indicators_count if metrics else 0),
                'is_suspected_app_fraud': to_numeric(metrics.is_suspected_app_fraud if metrics else False),
                'is_suspected_takeover': to_numeric(metrics.is_suspected_takeover if metrics else False),
                'is_account_takeover': to_numeric(metrics.is_account_takeover if metrics else False),
                'multi_indicator_present': to_numeric(metrics.multi_indicator_present if metrics else False),
                
                # Transaction count and volume metrics
                'avg_monthly_transaction_count': to_numeric(metrics.avg_monthly_transaction_count if metrics else 0),
                'avg_monthly_transaction_volume': to_numeric(metrics.avg_monthly_transaction_volume if metrics else 0),
                'transaction_frequency_stddev': to_numeric(metrics.transaction_frequency_stddev if metrics else 0),
                
                # Additional existing metrics
                'tx_count_1h': to_numeric(metrics.velocity_24h / 24 if metrics else 0),
                'tx_count_24h': to_numeric(metrics.velocity_24h if metrics else 0),
                'route_count_1h': to_numeric(metrics.unique_recipients_24h / 24 if metrics else 0),
                'route_count_24h': to_numeric(metrics.unique_recipients_24h if metrics else 0),
            
                'hour': float(local_hour)
            }

            # Add channel-specific features
            if payment_channel == 'ZELLE':
                # ZELLE-specific features
                features['is_high_value_zelle'] = 1.0 if transaction.get('amount', 0) >  high_amount_threshold else 0.0
                features['zelle_daily_limit_ratio'] = min(metrics.amount_24h / float(self.config_reader.get_property('zelle.threshold.daily_limit', '3000')), 1.0)
                features['zelle_new_recipient'] = 1.0 if metrics.new_recipient else 0.0
                features['zelle_recipients_24h'] = min(metrics.unique_recipients_24h / float(self.config_reader.get_property('zelle.threshold.max_recipients', '5')), 1.0)
                features['zelle_after_hours'] = to_numeric(local_hour < 6 or local_hour > 23) #UTC check
                features['zelle_weekend'] = to_numeric(is_weekend) #UTC Check
                features['zelle_amount_deviation'] = min(abs((transaction.get('amount', 0) - metrics.avg_amount_30d) / max(metrics.std_amount_30d, 1.0)), 5.0) if metrics.std_amount_30d > 0 else 3.0
                features['zelle_velocity_ratio'] = min(metrics.velocity_24h / float(self.config_reader.get_property('zelle.threshold.velocity_24h', '6')), 1.0)
            elif payment_channel == 'ACH':
                # ACH-specific features
                features['is_high_value_ach'] = 1.0 if transaction.get('amount', 0) > float(self.config_reader.get_property('ach.threshold.high_amount', '10000')) else 0.0
                features['ach_same_day'] = 1.0 if transaction.get('same_day', False) else 0.0
                features['ach_return_risk'] = 1.0 if transaction.get('return_code') in ['R01', 'R02', 'R03', 'R04'] else 0.0
                features['ach_off_hours'] = 1.0 if datetime.now().hour < 8 or datetime.now().hour > 18 else 0.0
                features['ach_weekend'] = 1.0 if datetime.now().weekday() >= 5 else 0.0
                features['ach_suspicious_odfi'] = 1.0 if transaction.get('odfi_id') in self.channel_detector.suspicious_odfi_list else 0.0
                features['ach_suspicious_rdfi'] = 1.0 if transaction.get('rdfi_id') in self.channel_detector.suspicious_rdfi_list else 0.0
                features['ach_batch_number'] = min(int(transaction.get('batch_number', 1)), 10) / 10  # Normalize to 0.1-1.0    
            elif payment_channel == 'WIRE':
                # WIRE-specific features
                wire_threshold_high_amount = float(self.config_reader.get_property('wire.threshold.high_amount', '50000'))
                features['is_high_value_wire'] = 1.0 if transaction.get('amount', 0) > wire_threshold_high_amount else 0.0
                features['wire_cross_border'] = 1.0 if transaction.get('is_cross_border', False) else 0.0
                features['wire_high_risk_country'] = 1.0 if transaction.get('involves_high_risk_country', False) else 0.0
                features['wire_business_hours'] = 1.0 if (9 <= datetime.now().hour <= 17) and (0 <= datetime.now().weekday() <= 4) else 0.0
                features['wire_amount_to_history_ratio'] = transaction.get('amount', 0) / max(metrics.avg_amount_30d, 1.0) if metrics.avg_amount_30d > 0 else 5.0
            elif payment_channel == 'SWIFT':
                # SWIFT-specific features
                swift_threshold_high_amount = float(self.config_reader.get_property('swift.threshold.high_amount', '100000'))
                features['is_high_value_swift'] = 1.0 if transaction.get('amount', 0) >  swift_threshold_high_amount else 0.0
                features['swift_cross_border'] = 1.0 if transaction.get('is_cross_border', False) else 0.0
                features['swift_high_risk_country'] = 1.0 if transaction.get('involves_high_risk_country', False) else 0.0
                features['swift_correspondent_bank'] = 1.0 if transaction.get('involves_correspondent', False) else 0.0
                features['swift_weekend_transaction'] = 1.0 if datetime.now().weekday() >= 5 else 0.0
                features['swift_nonstandard_message_type'] = 1.0 if transaction.get('message_type', '') not in ['MT103', 'MT202'] else 0.0
            elif payment_channel == 'FEDNOW':
                # FEDNOW-specific features
                features['is_high_value_fednow'] = 1.0 if transaction.get('amount', 0) > float(self.config_reader.get_property('fednow.threshold.high_amount', '25000')) else 0.0
                features['fednow_rapid_succession'] = 1.0 if metrics.velocity_24h > float(self.config_reader.get_property('fednow.threshold.velocity_24h', '3')) else 0.0
                features['fednow_off_hours'] = 1.0 if datetime.now().hour < 6 or datetime.now().hour > 22 else 0.0
                features['fednow_new_recipient'] = 1.0 if metrics.new_recipient else 0.0
                features['fednow_amount_deviation'] = abs((transaction.get('amount', 0) - metrics.avg_amount_30d) / max(metrics.std_amount_30d, 1.0)) if metrics.std_amount_30d > 0 else 3.0
                features['fednow_instant_flag'] = 1.0  # FedNow is always instant
                

            # Create DataFrame and encode features
            df = pd.DataFrame([features])
            encoded_df = self._encode_categorical(df)
            
            # Ensure all values are numeric
            encoded_df = encoded_df.astype(float)
            
            # Stack features
            combined = encoded_df.values
            
            # logger.debug(f"Extracted features shape: {combined.shape}, dtype: {combined.dtype}")
            return combined
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return np.zeros((1, len(self.feature_names)), dtype=float)
        
    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features using fixed categories"""
        try:

            encoded_df = df.copy()
            encoded_columns = []
        
            # Process each categorical feature
            for feature in self.categorical_features:
                # Create binary columns for each known category
                for category in self.fixed_categories[feature]:
                    col_name = f"{feature}_{category}"
                    encoded_df[col_name] = (df[feature] == category).astype(float)
                    encoded_columns.append(col_name)
                
                # Drop original column
                encoded_df = encoded_df.drop(feature, axis=1)
        
            # Store feature names if not already set
            if not self.feature_names:
                self.feature_names = self.numerical_features + encoded_columns
                # logger.info(f"Feature names set, total features: {len(self.feature_names)}")
                
            return encoded_df[self.feature_names]  # Return only the expected feature
        except Exception as e:
            logger.error(f"Error in categorical encoding: {str(e)}")
            raise

    def prepare_training_data(self, transactions: List[Dict], labels: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and labels for training"""
        # logger.info("Starting prepare_training_data...")
        try:
            # First fit the feature extractor (self)
            # logger.info("Fitting feature extractor...")
            self.fit(transactions)
            # logger.info("Feature extractor fitted")
            
            features_list = []
            valid_indices = []
            
            # Create metrics for each transaction
            for i, transaction in enumerate(transactions):
                # logger.info(f"Processing transaction {i+1}/{len(transactions)}")
                
                # Create metrics
                metrics = TransactionMetrics(
                    velocity_24h=0,
                    amount_24h=0,
                    unique_recipients_24h=0,
                    velocity_7d=0,
                    amount_7d=0,
                    unique_recipients_7d=0,
                    avg_amount_30d=0,
                    std_amount_30d=0,
                    new_recipient=False,
                    cross_border=transaction.get('is_cross_border', False),
                    high_risk_country=transaction.get('involves_high_risk_country', False)
                )
                
                try:
                    # Extract features using self (this instance is the feature extractor)
                    features = self.extract_features(transaction, metrics)
                    features_list.append(features)
                    valid_indices.append(i)
                except Exception as e:
                    logger.error(f"Error extracting features for transaction {i}: {str(e)}")
                    continue
            
            if not features_list:
                raise ValueError("No features could be extracted from transactions")
                
            X = np.vstack(features_list)
            y = np.array([labels[i] for i in valid_indices])
            
            # logger.info(f"Final shapes - X: {X.shape}, y: {y.shape}")
            
            if len(y) != X.shape[0]:
                raise ValueError(f"Number of labels ({len(y)}) does not match number of feature vectors ({X.shape[0]})")
                
            return X, y
            
        except Exception as e:
            logger.error(f"Error in prepare_training_data: {str(e)}")
            raise

class MLModel:
    """Base class for ML models"""
    def _ensure_binary_labels(self, y: np.ndarray) -> np.ndarray:
        """Ensure labels are binary"""
        unique_labels = np.unique(y)
        if len(unique_labels) == 1:
            # If only one class, create synthetic data for the other class
            if unique_labels[0] == 0:
                return np.append(y, [1])  # Add a positive example
            else:
                return np.append(y, [0])  # Add a negative example
        return y
    
    def predict_proba(self, features: np.ndarray) -> float:
        try:
            # Convert all values to float and handle special cases
            def convert_to_float(value):
                if isinstance(value, (int, float)):
                    return float(value)
                elif isinstance(value, bool):
                    return 1.0 if value else 0.0
                elif isinstance(value, str):
                    # Convert string booleans
                    if value.lower() == 'true':
                        return 1.0
                    elif value.lower() == 'false':
                        return 0.0
                    # Try to convert numeric strings
                    try:
                        return float(value)
                    except ValueError:
                        return 0.0
                return 0.0
            # Handle dictionary input
            if isinstance(features, dict):
                features_array = np.array([convert_to_float(v) for v in features.values()]).reshape(1, -1)
            # Handle numpy array input
            elif isinstance(features, np.ndarray):
                # Convert each element to float
                features_array = np.vectorize(convert_to_float)(features)
                if len(features_array.shape) == 1:
                    features_array = features_array.reshape(1, -1)
            else:
                logger.error(f"Unsupported features type: {type(features)}")
                return 0.0
            
            # Ensure correct number of features
            expected_features = self.model.n_features_in_
            current_features = features_array.shape[1]
            
            if current_features != expected_features:
                #logger.warning(f"Adjusting feature count from {current_features} to {expected_features}")
                if current_features < expected_features:
                    # Pad with zeros
                    features_array = np.pad(
                        features_array, 
                        ((0, 0), (0, expected_features - current_features)),
                        'constant',
                        constant_values=0
                    )
                else:
                    # Truncate
                    features_array = features_array[:, :expected_features]

            # Handle NaN and inf values
            features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)

            # Verify final array
            if not np.issubdtype(features_array.dtype, np.number):
                features_array = features_array.astype(float)

            # logger.debug(f"Final features shape: {features_array.shape}, dtype: {features_array.dtype}")
            return self._predict_proba_impl(features_array)

        except Exception as e:
            logger.error(f"Error in predict_proba: {str(e)}")
            return 0.0

    
    def _predict_proba_impl(self, features: np.ndarray) -> float:
        raise NotImplementedError
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        raise NotImplementedError

class RandomForestModel(MLModel):
    def __init__(self, n_estimators=100):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        self.n_features = None
        
    def _predict_proba_impl(self, features: np.ndarray) -> float:
        try:
            if not hasattr(self.model, 'n_features_in_'):
                logger.error("Model not fitted yet")
                return 0.0
                
            probs = self.model.predict_proba(features)
            return float(probs[0, 1]) if probs.shape[1] >= 2 else 0.0
            
        except Exception as e:
            logger.error(f"Error in RandomForest prediction: {str(e)}")
            return 0.0
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        # Store number of features
        self.n_features = X.shape[1]
        
        # Ensure binary labels
        y = self._ensure_binary_labels(y)
        
        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0)
        
        self.model.fit(X, y)

class NeuralNetModel(MLModel):
    def __init__(self):
        self.model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            random_state=42
        )
        self.n_features = None
        
    def _predict_proba_impl(self, features: np.ndarray) -> float:
        try:
            if not hasattr(self.model, 'n_features_in_'):
                logger.error("Model not fitted yet")
                return 0.0
                
            probs = self.model.predict_proba(features)
            return float(probs[0, 1]) if probs.shape[1] >= 2 else 0.0
            
        except Exception as e:
            logger.error(f"Error in RandomForest prediction: {str(e)}")
            return 0.0
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        # Store number of features
        self.n_features = X.shape[1]
        
        # Ensure binary labels
        y = self._ensure_binary_labels(y)
        
        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0)
        
        self.model.fit(X, y)

class XGBoostModel(MLModel):
    def __init__(self):
        self.model = xgb.XGBClassifier(
            max_depth=6,
            learning_rate=0.1,
            n_estimators=100,
            objective='binary:logistic',
            random_state=42
        )
        self.n_features = None
        
    def _predict_proba_impl(self, features: np.ndarray) -> float:
        try:
            if not hasattr(self.model, 'n_features_in_'):
                logger.error("Model not fitted yet")
                return 0.0
                
            probs = self.model.predict_proba(features)
            return float(probs[0, 1]) if probs.shape[1] >= 2 else 0.0
            
        except Exception as e:
            logger.error(f"Error in RandomForest prediction: {str(e)}")
            return 0.0
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        # Store number of features
        self.n_features = X.shape[1]
        
        # Ensure binary labels
        y = self._ensure_binary_labels(y)
        
        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0)
        
        self.model.fit(X, y)

class EnsembleFraudDetectionModel:
    """Ensemble model combining multiple ML approaches"""
    
    def __init__(self, config_reader: ConfigReader,db_connection: CSVDatabaseConnection = None, model_weights: Dict[str, float] = None):
        if db_connection is None:
        # Create a new CSV database connection if none is provided
            self.db = CSVDatabaseConnection(config_path=config_reader.config_file_path)
        else:
            self.db = db_connection
        
        self.config_reader = config_reader
        self.feature_extractor = TransactionFeatureExtractor()
        self.analyzer = TransactionAnalyzer()
        # self.pattern_detector = FraudPatternDetector()
        self.channel_detector = ChannelAlertDetector(config_reader, db_connection=self.db)
        
        
        self.model_weights = model_weights or {
            'random_forest': 0.3,
            'neural_net': 0.3,
            'xgboost': 0.4
        }
        self.models = {
            'random_forest': RandomForestModel(),
            'neural_net': NeuralNetModel(),
            'xgboost': XGBoostModel()
        }
        
        self.risk_thresholds = self._load_risk_thresholds()
        self.risk_weights = {
            'ml_score': float(self.config_reader.get_property('risk.weightage_check.ml_score', '0.25')),
            'rule_based': float(self.config_reader.get_property('risk.weightage_check.rule_based', '0.25')),
            'pattern_based': float(self.config_reader.get_property('risk.weightage_check.pattern_based', '0.15')),
            'channel_specific': float(self.config_reader.get_property('risk.weightage_check.channel_specific', '0.15')),
            'app_fraud': float(self.config_reader.get_property('risk.weightage_check.app_fraud', '0.20'))  # New weight
        }
    
        # Initialize feature extractor and train models with sample data
        self._initialize_models()

    def _initialize_models(self):
        # print(f"DEBUG: _initialize_models called")
        # print(f"DEBUG: self.config_reader is {type(self.config_reader)}")
        # print(f"DEBUG: self.feature_extractor is {type(self.feature_extractor)}")
        # print(f"DEBUG: self.feature_extractor.config_reader is {type(self.feature_extractor.config_reader)}")
    
        """Initialize and train models with sample data"""
        try:
            # Create sample training data
            sample_transactions = [
                {
                    'amount': 1000.0,
                    'channel': 'WIRE',
                    'currency': 'USD',
                    'debtor_country': 'US',
                    'creditor_country': 'GB',
                    'is_cross_border': True,
                    'involves_high_risk_country': False
                },
                {
                    'amount': 500.0,
                    'channel': 'ACH',
                    'currency': 'USD',
                    'debtor_country': 'US',
                    'creditor_country': 'US',
                    'is_cross_border': False,
                    'involves_high_risk_country': False
                },
                {
                    'amount': 5000.0,
                    'channel': 'WIRE',
                    'currency': 'USD',
                    'debtor_country': 'US',
                    'creditor_country': 'IR',
                    'is_cross_border': True,
                    'involves_high_risk_country': True
                }
            ]
            
            # Create sample labels (0 for non-fraudulent, 1 for fraudulent)
            sample_labels = [0, 0, 1]  # Both transactions are non-fraudulent
            
            # Initialize feature extractor
            # print(f"DEBUG: About to fit feature_extractor with sample data")
            self.feature_extractor.fit(sample_transactions)
            # logger.info("Feature extractor initialized successfully")
            # print(f"DEBUG: feature_extractor.fit completed")

            # Create metrics for sample transactions
            metrics_list = []
            features_list = []
            valid_indices = []

            # Process each transaction
            for i, transaction in enumerate(sample_transactions):
                try:
                    metrics = TransactionMetrics(
                        velocity_24h=0,
                        amount_24h=0,
                        unique_recipients_24h=0,
                        velocity_7d=0,
                        amount_7d=0,
                        unique_recipients_7d=0,
                        avg_amount_30d=0,
                        std_amount_30d=0,
                        new_recipient=False,
                        cross_border=transaction['is_cross_border'],
                        high_risk_country=transaction['involves_high_risk_country']
                    )
                    
                    features = self.feature_extractor.extract_features(transaction, metrics)
                    features_list.append(features)
                    metrics_list.append(metrics)
                    valid_indices.append(i)
                    
                except Exception as e:
                    # print(f"DEBUG: Exception in _initialize_models: {str(e)}")
                    # print(f"DEBUG: Traceback: {traceback.format_exc()}")
                    logger.error(f"Error processing sample transaction {i}: {str(e)}")
                    continue
            
            if not features_list:
                raise ValueError("No valid features could be extracted from sample transactions")
            
            # Stack features and get corresponding labels
            X = np.vstack(features_list)
            y = np.array([sample_labels[i] for i in valid_indices])
            
            # logger.info(f"Prepared training data - X shape: {X.shape}, y shape: {y.shape}")
            
            if len(y) != X.shape[0]:
                raise ValueError(f"Feature and label count mismatch: {X.shape[0]} features vs {len(y)} labels")
            
            # Train each model
            for name, model in self.models.items():
                try:
                    model.fit(X, y)
                    # logger.info(f"Initialized and trained {name} model")
                except Exception as e:
                    logger.error(f"Error training {name} model: {str(e)}")
                    raise
            
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise    

    def _load_risk_thresholds(self) -> Dict:
        """
        Load risk thresholds from config.properties
        Returns a dictionary with default values if not found in config
        """
        try:
            return {
                'velocity_24h': int(self.config_reader.get_property('risk.threshold.velocity_24h', '5')),
                'amount_24h': float(self.config_reader.get_property('risk.threshold.amount_24h', '10000')),
                'unique_recipients_24h': int(self.config_reader.get_property('risk.threshold.unique_recipients_24h', '3')),
                'amount_deviation': float(self.config_reader.get_property('risk.threshold.amount_deviation', '2.0')),
                'fraud_score': int(self.config_reader.get_property('risk.threshold.fraud_score', '60'))
            }
        except Exception as e:
            self.logger.error(f"Error loading risk thresholds from config: {str(e)}")
            # Return defaults if there's an error
            return {
                'velocity_24h': 5,
                'amount_24h': 10000,
                'unique_recipients_24h': 3,
                'amount_deviation': 2.0,
                'fraud_score': 60
            }

    def calculate_consolidated_risk(self, features):
        """Calculate consolidated risk score and analysis."""

        #########Commented out as compliance alert is being raised seperately as per Alak 3-Mar-2025#########
        # Add compliance risk components
        # compliance_risk = 0.0
        # compliance_indicators = {
        #     'ofac_match': features.get('ofac_match', False),
        #     'watchlist_match': features.get('watchlist_match', False),
        #     'high_compliance_risk': features.get('ofac_match', False) or features.get('watchlist_match', False)
        # }
        
        # if compliance_indicators['ofac_match']:
        #     compliance_risk += float(self.config_reader.get_property('ofac.risk_score', '100')) * \
        #                      float(self.config_reader.get_property('ofac.severity_multiplier', '1.5'))
        
        # if compliance_indicators['watchlist_match']:
        #     compliance_risk += float(self.config_reader.get_property('watchlist.risk_score', '100')) * \
        #                      float(self.config_reader.get_property('watchlist.severity_multiplier', '1.2'))
        ####################################################################################

        # Base risk components with enhanced scoring
        amount_risk = float(features['amount_zscore'])
        velocity_risk = float(features['tx_count_1h'] + 
                     features['route_count_1h'] * float(self.config_reader.get_property('risk.weight.route_count', '1.5')))
        cross_border_risk = float(features['cross_border'] * float(self.config_reader.get_property('risk.weight.cross_border', '1.2')) + 
                         features['high_risk_route'] * float(self.config_reader.get_property('risk.weight.high_risk_route', '1.8')))

        #  BEHAVIORAL CONSISTENCY RISK COMPONENT
        behavioral_consistency_risk = 0.0
        if 'behavioral_consistency_score' in features:
            consistency_threshold = float(self.config_reader.get_property('risk.threshold.behavioral_consistency', '60'))
            if float(features['behavioral_consistency_score']) < consistency_threshold:
                behavioral_consistency_risk = (consistency_threshold - float(features['behavioral_consistency_score'])) / consistency_threshold * 100
        
        #  TRANSACTION PREDICTABILITY RISK COMPONENT
        predictability_risk = 0.0
        if 'transaction_predictability_score' in features:
            predictability_threshold = float(self.config_reader.get_property('risk.threshold.predictability', '50'))
            if float(features['transaction_predictability_score']) < predictability_threshold:
                predictability_risk = (predictability_threshold - float(features['transaction_predictability_score'])) / predictability_threshold * 100
        
        #  ACCOUNT MATURITY RISK COMPONENT
        account_maturity_risk = 0.0
        if 'account_age_days' in features:
            new_account_threshold = int(self.config_reader.get_property('risk.threshold.account_age_days', '90'))
            if int(features['account_age_days']) < new_account_threshold:
                account_maturity_risk = (new_account_threshold - int(features['account_age_days'])) / new_account_threshold * 70
        
        #  PATTERN BREAK RISK COMPONENT
        pattern_break_risk = 0.0
        if 'pattern_break_frequency_365d' in features:
            pattern_break_threshold = int(self.config_reader.get_property('risk.threshold.pattern_break_frequency', '3'))
            if int(features['pattern_break_frequency_365d']) > pattern_break_threshold:
                pattern_break_risk = (int(features['pattern_break_frequency_365d']) - pattern_break_threshold) * 15
        
        #  SEASONAL PATTERN DEVIATION RISK COMPONENT
        seasonal_risk = 0.0
        if 'seasonal_pattern_deviation' in features:
            seasonal_threshold = float(self.config_reader.get_property('risk.threshold.seasonal_deviation', '0.3'))
            if float(features['seasonal_pattern_deviation']) > seasonal_threshold:
                seasonal_risk = (float(features['seasonal_pattern_deviation']) - seasonal_threshold) / (1 - seasonal_threshold) * 70

        #UTC Check
        # Get country code
        country_code = features.get('debtor_country', 'UNKNOWN')
        
        # Get current time and convert to local time for risk assessment
        current_time = datetime.now(timezone.utc)
        
        # Calculate time risk with time zone awareness
        if hasattr(self, 'timezone_manager') and country_code != 'UNKNOWN':
            try:
                local_time = self.timezone_manager.convert_to_local_time(current_time, country_code)
                # Get business hours config for this country
                # Default business hours configuration
                business_config = {
                    'start_hour': 9,
                    'end_hour': 17,
                    'business_days': [0, 1, 2, 3, 4],  # Monday to Friday
                    'weekend_days': [5, 6]  # Saturday and Sunday
                }
            
                # Check if within business hours in local time
                hour = local_time.hour
                weekday = local_time.weekday()
                is_business_hour = (
                    business_config['start_hour'] <= hour < business_config['end_hour'] and
                    weekday in business_config['business_days']
                )
            
                time_risk = 0.0
                if not is_business_hour:
                    # Different risk for evening vs. middle of night
                    if hour < 6 or hour >= 22:  # Late night/early morning
                        time_risk = float(self.config_reader.get_property('risk.time.late_night', '0.8'))
                    else:  # Evening or early morning
                        time_risk = float(self.config_reader.get_property('risk.time.after_hours', '0.5'))
            
                # Add weekend risk if applicable
                business_hours_risk = 0.0
                if weekday in business_config['weekend_days']:
                    business_hours_risk = float(self.config_reader.get_property('risk.time.weekend', '0.6'))
            except Exception as e:
                logger.error(f"Error in timezone conversion: {str(e)}")
                # Fall back to feature-based approach
                time_risk = float(features['is_high_risk_hour'])
                business_hours_risk = float(1 - features['is_business_hours']) * float(self.config_reader.get_property('risk.weight.non_business_hours', '0.5'))
        else:
            # Fallback to simple feature-based approach
            time_risk = float(features['is_high_risk_hour'])
            business_hours_risk = float(1 - features['is_business_hours']) * float(self.config_reader.get_property('risk.weight.non_business_hours', '0.5'))

        # Detect potential account takeover patterns
        account_takeover_indicators = {
            'unusual_timing': features['is_high_risk_hour'] and not features['is_business_hours'],
            'velocity_spike': features['tx_count_1h'] > float(self.config_reader.get_property('account_takeover.threshold.hourly_tx_count', '3')) or features['route_count_1h'] > int(self.config_reader.get_property('account_takeover.threshold.hourly_route_count', '2')),
            'high_amount': abs(features['amount_zscore']) > float(self.config_reader.get_property('account_takeover.threshold.amount_zscore', '2.5')),
            'unusual_destination': features['cross_border'] and features['high_risk_route'],
            'rapid_succession': features['tx_count_1h'] / (features['tx_count_24h'] + 1) > float(self.config_reader.get_property('account_takeover.threshold.rapid_succession_ratio', '0.5'))
        }
        
        # Detect potential mule account patterns
        
        # mule_account_indicators = {
        #     'layering_pattern': features['route_count_24h'] > float(self.config_reader.get_property('mule_account.threshold.daily_route_count', '5')) and features['cross_border'],
        #     'systematic_amounts': abs(1 - (features['amount'] % 1000) / 1000) < float(self.config_reader.get_property('mule_account.threshold.systematic_amount_margin', '0.1')),
        #     'high_risk_corridor': features['high_risk_route'] and features['cross_border'],
        #     'rapid_outflow': features['tx_count_24h'] > float(self.config_reader.get_property('mule_account.threshold.daily_tx_count', '10')) and features['amount_zscore'] > float(self.config_reader.get_property('mule_account.threshold.amount_zscore', '1.5')),
        #     'new_beneficiaries': features['route_count_24h'] / (features['tx_count_24h'] + 1) > float(self.config_reader.get_property('mule_account.threshold.new_beneficiaries_ratio', '0.8')),
        
        #     # New indicators for the specific A-1234 to B-5678 pattern
        #     'multiple_diverse_sources': features.get('unique_sources_7d', 0) > float(self.config_reader.get_property('mule_account.threshold.diverse_sources', '10')),
        #     'off_hours_deposits': features.get('off_hours_txn_ratio', 0) > float(self.config_reader.get_property('mule_account.threshold.off_hours_ratio', '0.4')),
        #     'rounded_amounts': features.get('rounded_amount_ratio', 0) > float(self.config_reader.get_property('mule_account.threshold.rounded_amounts_ratio', '0.7')),
        #     'rapid_withdrawal': features.get('min_time_to_withdrawal_minutes', 999) < float(self.config_reader.get_property('mule_account.threshold.withdrawal_time_minutes', '120')),
        #     'account_emptying': features.get('max_withdrawal_pct', 0) > float(self.config_reader.get_property('mule_account.threshold.emptying_percentage', '85')),
        #     'geographic_diversity': features.get('geographic_diversity', 0) > float(self.config_reader.get_property('mule_account.threshold.geographic_diversity', '5')),
        #     'new_account_activity': features.get('account_age_days', 999) < float(self.config_reader.get_property('mule_account.threshold.account_age_days', '180')) and features.get('tx_count_7d', 0) > 10,
        #     'funnel_pattern': features.get('unique_sources_7d', 0) > 5 and features.get('unique_destinations_7d', 0) <= 2
        
        # }

        # # Calculate score for potential mule relationship (A-1234 to B-5678 pattern)
        # mule_relationship_indicators = {
        #     'mule_to_recipient_pattern': features.get('is_suspected_mule', False) and features.get('recipient_risk_score', 0) > 50,
        #     'collect_and_transfer': features.get('rapid_collect_transfer', False),
        #     'first_time_recipient': features.get('is_new_recipient', False) and features['amount'] > 1000
        # }
        
        # Calculate specific risk scores for ATO and mule patterns
        ato_risk_score = sum(account_takeover_indicators.values()) * float(self.config_reader.get_property('risk.scale.account_takeover', '20'))  # Scale to 0-100
        # mule_risk_score = sum(mule_account_indicators.values()) * float(self.config_reader.get_property('risk.scale.mule_account', '20'))     # Scale to 0-100
        # relationship_risk_score = sum(mule_relationship_indicators.values()) * float(self.config_reader.get_property('risk.scale.mule_relationship', '33.3'))  # Scale to 0-100 (with 3 indicators)
    
        # Enhanced risk weights with ATO and mule components
        try:
            risk_weights = {
                'amount': float(self.config_reader.get_property('risk.weight.factor.amount', '0.20')),
                'velocity': float(self.config_reader.get_property('risk.weight.factor.velocity', '0.15')),
                'cross_border': float(self.config_reader.get_property('risk.weight.factor.cross_border', '0.15')),
                'time': float(self.config_reader.get_property('risk.weight.factor.time', '0.10')),
                'business_hours': float(self.config_reader.get_property('risk.weight.factor.business_hours', '0.05')),
                'account_takeover': float(self.config_reader.get_property('risk.weight.factor.account_takeover', '0.20'))
                # 'mule_account': float(self.config_reader.get_property('risk.weight.factor.mule_account', '0.15')),
                # 'mule_relationship': float(self.config_reader.get_property('risk.weight.factor.mule_relationship', '0.15'))
       
            }
        except Exception as e:
            self.logger.error(f"Error loading risk factor weights from config: {str(e)}")
            # Return defaults if there's an error
            risk_weights = {
                'amount': 0.15,
                'velocity': 0.10,
                'cross_border': 0.15,
                'time': 0.10,
                'business_hours': 0.05,
                'account_takeover': 0.15
                # 'mule_account': 0.15,
                # 'mule_relationship': 0.15
            }

        
        risk_components = {
            ##Commented out as compliance alerts are being handled seperately as per Alak on 3-Mar-2025##
            # 'compliance_risk' : {
            #     'score': min(compliance_risk, float(self.config_reader.get_property('compliance.max_risk_score', '100'))),
            #     'weight': float(self.config_reader.get_property('risk.weight.factor.compliance', '0.25')),
            #     'factors': [factor for factor, is_present in {
            #         'OFAC SDN List Match': compliance_indicators['ofac_match'],
            #         'Watchlist Match': compliance_indicators['watchlist_match']
            #     }.items() if is_present],
            #     'raw_score': compliance_risk
            # },
            ################################################################
            'amount_risk': {
                'score': min(max(amount_risk * float(self.config_reader.get_property('risk.score.scale.amount', '20')), 0), 100),
                'weight': risk_weights['amount'],
                'factors': [
                    'Unusual transaction amount',
                    'Historical pattern deviation',
                    'Amount zscore: {:.2f}'.format(amount_risk)
                ],
                'raw_score': amount_risk
            },
            'velocity_risk': {
                'score': min(max(velocity_risk * float(self.config_reader.get_property('risk.score.scale.velocity', '15')), 0), 100),
                'weight': risk_weights['velocity'],
                'factors': [
                    'High transaction frequency',
                    'Unusual activity pattern',
                    'Hourly transactions: {}'.format(features['tx_count_1h']),
                    'Route frequency: {}'.format(features['route_count_1h'])
                ],
                'raw_score': velocity_risk
            },
            'cross_border_risk': {
                'score': min(max(cross_border_risk * float(self.config_reader.get_property('risk.score.scale.cross_border', '50')), 0), 100),
                'weight': risk_weights['cross_border'],
                'factors': [
                    'Cross-border transaction' if features['cross_border'] else None,
                    'High-risk country involved' if features['high_risk_route'] else None,
                    'Countries: {} -> {}'.format(features['debtor_country'], features['creditor_country'])
                ],
                'raw_score': cross_border_risk
            },
            'account_takeover_risk': {
                'score': min(max(ato_risk_score, 0), 100),
                'weight': risk_weights['account_takeover'],
                'factors': [factor for factor, is_present in {
                    'Suspicious after-hours activity': account_takeover_indicators['unusual_timing'],
                    'Sudden transaction velocity spike': account_takeover_indicators['velocity_spike'],
                    'Abnormal transaction amount': account_takeover_indicators['high_amount'],
                    'Unusual destination account': account_takeover_indicators['unusual_destination'],
                    'Rapid transaction succession': account_takeover_indicators['rapid_succession']
                }.items() if is_present],
                'raw_score': float(ato_risk_score)
            },
            # 'mule_account_risk': {
            #     'score': min(max(mule_risk_score, 0), 100),
            #     'weight': risk_weights['mule_account'],
            #     'factors': [factor for factor, is_present in {
            #         'Transaction layering pattern': mule_account_indicators['layering_pattern'],
            #         'Round amount transactions': mule_account_indicators['systematic_amounts'],
            #         'High-risk corridor activity': mule_account_indicators['high_risk_corridor'],
            #         'Rapid funds outflow': mule_account_indicators['rapid_outflow'],
            #         'Multiple new beneficiaries': mule_account_indicators['new_beneficiaries'],
            #         'Multiple diverse deposit sources': mule_account_indicators['multiple_diverse_sources'],
            #         'Off-hours deposit activity': mule_account_indicators['off_hours_deposits'],
            #         'Pattern of rounded amounts': mule_account_indicators['rounded_amounts'],
            #         'Rapid withdrawal after deposits': mule_account_indicators['rapid_withdrawal'],
            #         'Account emptying pattern': mule_account_indicators['account_emptying'],
            #         'Geographically diverse sources': mule_account_indicators['geographic_diversity'],
            #         'New account with high activity': mule_account_indicators['new_account_activity'],
            #         'Many-to-one funnel pattern': mule_account_indicators['funnel_pattern']
            #     }.items() if is_present],
            #     'raw_score': float(mule_risk_score)
            # },
            # 'mule_relationship_risk': {
            #     'score': min(max(relationship_risk_score, 0), 100),
            #     'weight': risk_weights['mule_relationship'],
            #     'factors': [factor for factor, is_present in {
            #         'Mule account to recipient pattern': mule_relationship_indicators['mule_to_recipient_pattern'],
            #         'Rapid collection and transfer': mule_relationship_indicators['collect_and_transfer'],
            #         'First-time large transfer to recipient': mule_relationship_indicators['first_time_recipient']
            #     }.items() if is_present],
            #     'raw_score': float(relationship_risk_score)
            # },
            'time_risk': {
                'score': min(max((time_risk + business_hours_risk) * float(self.config_reader.get_property('risk.score.scale.time', '70')), 0), 100),
                'weight': risk_weights['time'] + risk_weights['business_hours'],
                'factors': [
                    'Transaction during high-risk hours' if time_risk > 0 else None,
                    'Outside business hours' if business_hours_risk > 0 else None,
                    'Hour of day: {}'.format(hour if 'hour' in locals() else features.get('hour', 'unknown'))
                ],
                'raw_score': time_risk + business_hours_risk
            },
            'behavioral_consistency_risk': {
                'score': min(max(behavioral_consistency_risk, 0), 100),
                'weight': float(self.config_reader.get_property('risk.weight.factor.behavioral_consistency', '0.15')),
                'factors': [
                    'Low behavioral consistency',
                    'Unpredictable transaction patterns',
                    f"Consistency score: {features.get('behavioral_consistency_score', 0)}"
                ] if behavioral_consistency_risk > 0 else [],
                'raw_score': behavioral_consistency_risk
            },
            
            'transaction_predictability_risk': {
                'score': min(max(predictability_risk, 0), 100),
                'weight': float(self.config_reader.get_property('risk.weight.factor.predictability', '0.10')),
                'factors': [
                    'Unpredictable transaction behavior',
                    f"Predictability score: {features.get('transaction_predictability_score', 0)}"
                ] if predictability_risk > 0 else [],
                'raw_score': predictability_risk
            },
            
            'account_maturity_risk': {
                'score': min(max(account_maturity_risk, 0), 100),
                'weight': float(self.config_reader.get_property('risk.weight.factor.account_maturity', '0.10')),
                'factors': [
                    'New or recently created account',
                    f"Account age: {features.get('account_age_days', 0)} days"
                ] if account_maturity_risk > 0 else [],
                'raw_score': account_maturity_risk
            },
            
            'pattern_break_risk': {
                'score': min(max(pattern_break_risk, 0), 100),
                'weight': float(self.config_reader.get_property('risk.weight.factor.pattern_break', '0.12')),
                'factors': [
                    'Frequent pattern breaks in behavior',
                    f"Pattern breaks: {features.get('pattern_break_frequency_365d', 0)}"
                ] if pattern_break_risk > 0 else [],
                'raw_score': pattern_break_risk
            },
            
            'seasonal_pattern_risk': {
                'score': min(max(seasonal_risk, 0), 100),
                'weight': float(self.config_reader.get_property('risk.weight.factor.seasonal_pattern', '0.08')),
                'factors': [
                    'Deviation from seasonal transaction patterns',
                    f"Seasonal deviation: {features.get('seasonal_pattern_deviation', 0)}"
                ] if seasonal_risk > 0 else [],
                'raw_score': seasonal_risk
            }
        }

        # Calculate consolidated score with exponential penalty
        base_score = sum(
            component['score'] * component['weight']
            for component in risk_components.values()
        )
        
        high_risk_penalty = sum(
            float(self.config_reader.get_property('risk.penalty.base_multiplier', '0.1')) * max(0, component['score'] - float(self.config_reader.get_property('risk.penalty.threshold', '80'))) ** float(self.config_reader.get_property('risk.penalty.exponent', '1.5'))
            for component in risk_components.values()
        )
        
        if ato_risk_score > float(self.config_reader.get_property('risk.combined.ato_threshold', '60')): 
        # and mule_risk_score > float(self.config_reader.get_property('risk.combined.mule_threshold', '60')):
            high_risk_penalty *= float(self.config_reader.get_property('risk.combined.penalty_multiplier', '1.5'))  # Additional penalty for combined ATO and mule indicators
        
        # # Additional penalty for strong mule relationship pattern
        # if relationship_risk_score > float(self.config_reader.get_property('risk.mule_relationship.threshold', '70')):
        #     high_risk_penalty *= float(self.config_reader.get_property('risk.mule_relationship.penalty_multiplier', '1.5'))
        
        consolidated_score = min(base_score + high_risk_penalty, 100)
        
        # Filter out None values and collect contributing factors
        contributing_factors = []
        for component, details in risk_components.items():

            factors = [f for f in details['factors'] if f is not None]
            
            if details['score'] > float(self.config_reader.get_property('risk.reporting.factor_score_threshold','50')):  # Only include high-risk factors
                contributing_factors.extend(factors)
        
        # Enhanced risk level classification
        risk_level = 'LOW' if consolidated_score < float(self.config_reader.get_property('risk.level.threshold.medium','50')) else \
                    'MEDIUM' if consolidated_score < float(self.config_reader.get_property('risk.level.threshold.high','70')) else \
                    'HIGH' if consolidated_score < float(self.config_reader.get_property('risk.level.threshold.critical','85')) else \
                    'CRITICAL'
        
        # Enhanced fraud type classification
        fraud_types = []
        
        # Account Takeover patterns
        
        if ato_risk_score > float(self.config_reader.get_property('account_takeover.threshold.high','60')):
            fraud_types.append(self.config_reader.get_property('account_takeover.threshold.high.fraud_type','POTENTIAL_ACCOUNT_TAKEOVER'))
        elif ato_risk_score > float(self.config_reader.get_property('account_takeover.threshold.medium','40')):
            fraud_types.append(self.config_reader.get_property('account_takeover.threshold.medium.fraud_type','SUSPICIOUS_ACCOUNT_ACTIVITY'))
            
        # # Mule Account patterns    
        # if mule_risk_score > float(self.config_reader.get_property('mule_account.threshold.high','60')):
        #     if mule_account_indicators.get('funnel_pattern', False) and mule_account_indicators.get('account_emptying', False):
        #         fraud_types.append(self.config_reader.get_property('mule_account.threshold.funnel.fraud_type','FUNNEL_MULE_ACCOUNT'))
        #     else:
        #         fraud_types.append(self.config_reader.get_property('mule_account.threshold.high.fraud_type','POTENTIAL_MULE_ACCOUNT'))
        # elif mule_risk_score > float(self.config_reader.get_property('mule_account.threshold.medium','40')):
        #     fraud_types.append(self.config_reader.get_property('mule_account.threshold.medium.fraud_type','SUSPICIOUS_MONEY_FLOW'))
        
        # # Specific A-1234 to B-5678 pattern detection
        # if relationship_risk_score > float(self.config_reader.get_property('mule_relationship.threshold.high','70')):
        #     fraud_types.append(self.config_reader.get_property('mule_relationship.threshold.high.fraud_type','MULE_ACCOUNT_NETWORK'))
        # elif relationship_risk_score > float(self.config_reader.get_property('mule_relationship.threshold.medium','50')):
        #     fraud_types.append(self.config_reader.get_property('mule_relationship.threshold.medium.fraud_type','SUSPICIOUS_ACCOUNT_RELATIONSHIP'))
            
            
        # Amount-based patterns
        if amount_risk > float(self.config_reader.get_property('amount_risk.threshold.high','2.5')):
            fraud_types.append(self.config_reader.get_property('amount_risk.threshold.high.fraud_type','UNUSUAL_TRANSACTION_AMOUNT'))
        elif amount_risk > float(self.config_reader.get_property('amount_risk.threshold.medium','1.5')):
            fraud_types.append(self.config_reader.get_property('amount_risk.threshold.high.fraud_type','ATYPICAL_TRANSACTION_VALUE'))
            
        # Velocity-based patterns
        if velocity_risk > float(self.config_reader.get_property('velocity_risk.threshold.high','3.0')):
            fraud_types.append(self.config_reader.get_property('velocity_risk.threshold.high.fraud_type','HIGH_VELOCITY_ALERT'))
        elif velocity_risk > float(self.config_reader.get_property('velocity_risk.threshold.medium','2.0')):
            fraud_types.append(self.config_reader.get_property('velocity_risk.threshold.medium.fraud_type','INCREASED_TRANSACTION_FREQUENCY'))
            
        # Cross-border patterns
        if features['high_risk_route']:
            fraud_types.append(self.config_reader.get_property('cross_border.threshold.high_risk_route.fraud_type','HIGH_RISK_CORRIDOR'))
        elif features['cross_border'] and cross_border_risk > float(self.config_reader.get_property('cross_border_risk.threshold.high','1.5')):
            fraud_types.append(self.config_reader.get_property('cross_border.threshold.high.fraud_type','UNUSUAL_CROSS_BORDER_PATTERN'))
            
        # Time-based patterns
        if time_risk > 0 and not features['is_business_hours']:
            fraud_types.append(self.config_reader.get_property('time.risk.fraud_type','OUT_OF_HOURS_ACTIVITY'))
            
        # If no specific patterns are identified but risk is elevated
        if not fraud_types and consolidated_score > float(self.config_reader.get_property('general_risk.threshold.minimum','50')):
            fraud_types.append(self.config_reader.get_property('general_risk.threshold.fraud_type','GENERAL_RISK_ALERT'))
            
        # Add severity prefix to fraud types based on risk level
        risk_prefix = {
            'CRITICAL': 'CRITICAL_',
            'HIGH': 'HIGH_RISK_',
            'MEDIUM': 'MEDIUM_RISK_',
            'LOW': 'LOW_RISK_'
        }.get(risk_level, '')
        
        if risk_prefix:
            fraud_types = [f"{risk_prefix}{fraud_type}" for fraud_type in fraud_types]

        # Collect contributing factors
        contributing_factors = []
        for component, details in risk_components.items():
            if details['score'] > float(self.config_reader.get_property('risk.reporting.factor_score_threshold','50')):  # Only include high-risk factors
                contributing_factors.extend([f for f in details['factors'] if f is not None])
        
        return {
            'consolidated_score': float(consolidated_score),
            'risk_level': risk_level,
            'fraud_types': fraud_types,
            # 'component_summaries':component_summaries,
            'risk_components': risk_components,
            'contributing_factors': contributing_factors,
            'base_score': float(base_score),
            'high_risk_penalty': float(high_risk_penalty),
            'ato_indicators_triggered': sum(account_takeover_indicators.values()),
            # 'mule_indicators_triggered': sum(mule_account_indicators.values()),
            # 'mule_relationship_indicators_triggered': sum(mule_relationship_indicators.values())   
        }

    
    def _load_risk_weights(self) -> Dict:
        try:
            return {
                'ml_score': float(self.config_reader.get_property('risk.weightage_check.ml_score', '0.3')),
                'rule_based': float(self.config_reader.get_property('risk.weightage_check.rule_based', '0.3')),
                'pattern_based': float(self.config_reader.get_property('risk.weightage_check.pattern_based', '0.2')),
                'channel_specific': float(self.config_reader.get_property('risk.weightage_check.channel_specific', '0.2')),
                'app_fraud': float(self.config_reader.get_property('risk.weightage_check.app_fraud', '0.15')),
                # 'mule_account': float(self.config_reader.get_property('risk.weightage_check.mule_account', '0.10')),
                'account_takeover': float(self.config_reader.get_property('risk.weightage_check.account_takeover', '0.10'))
                # 'mule_relationship':float(self.config_reader.get_property('risk.weightage_check.mule_relationship', '0.15'))
            }
        except Exception as e:
            logger.error(f"Error loading risk weights from config: {str(e)}")
            # Return defaults if there's an error
            return {
                'ml_score': 0.25,
                'rule_based': 0.25,
                'channel_specific': 0.15,
                'app_fraud': 0.20,
                # 'mule_account': 0.20, 
                'account_takeover': 0.20
                # 'mule_relationship': 0.25
            }

    def get_ensemble_prediction(self, features: np.ndarray) -> float:
        try:
            # Ensure features is 2D
            if len(features.shape) == 1:
                features = features.reshape(1, -1)

            predictions = {}
            prediction_values = []
            valid_models = 0
            total_weight = 0.0
                
            # Get predictions from each model
            for model_name, model in self.models.items():
                try:
                    pred = model.predict_proba(features)
                    if isinstance(pred, (int, float)):
                        predictions[model_name] = pred
                        # print(f"Model is {model} and prediction is {pred}")
                        prediction_values.append(pred)
                        valid_models += 1
                    elif hasattr(pred, 'shape') and len(pred.shape) > 1 and pred.shape[1] > 1:
                        # For models returning probability array [p_class0, p_class1, ...], take p_class1
                        predictions[model_name] = pred[0, 1]  # Probability of positive class
                        prediction_values.append(pred[0, 1])
                        valid_models += 1
                    else:
                        # logger.warning(f"Unexpected prediction type from {model_name}: {type(pred)}")
                        predictions[model_name] = 0.0
                    
                    total_weight += self.model_weights[model_name]
                    
                except Exception as e:
                    logger.error(f"Error getting prediction from {model_name}: {str(e)}")
                    predictions[model_name] = 0.0

            # If all predictions failed, return safe default
            if total_weight == 0:
                # logger.warning("All model predictions failed, returning default score")
                return {
                    'score': 0.0,
                    'confidence': 0.0,
                    'fraud_types': []
                }
            
            # Calculate weighted average, adjusting weights if some models failed
            ensemble_score = sum(
                predictions[model_name] * (self.model_weights[model_name] / total_weight)
                for model_name in predictions.keys()
            )

            # Calculate confidence metrics
            confidence = 0.0
            
            if valid_models > 0:
                # 1. Model agreement - how consistent are the predictions across models?
                if valid_models > 1:
                    prediction_std = np.std(prediction_values)
                    # Convert standard deviation to a confidence metric (lower std = higher confidence)
                    # We want 0 std to be 1.0 confidence, and high std to be low confidence
                    model_agreement = max(0.0, 1.0 - (prediction_std * 2.0))
                else:
                    model_agreement = 0.7  # Default for single model
                
                # 2. Prediction extremity - scores near 0 or 1 typically have higher confidence
                # For binary classification, prediction closer to 0 or 1 indicates more confidence
                prediction_extremity = 2.0 * abs(ensemble_score - 0.5)
                
                # 3. Valid model ratio - more working models = higher confidence
                model_ratio = valid_models / max(1, len(self.models))
                
                # Combine factors - customize these weights based on your specific needs
                confidence = (
                    (0.5 * model_agreement) + 
                    (0.3 * prediction_extremity) + 
                    (0.2 * model_ratio)
                )
                
                # Ensure confidence is between 0 and 1
                confidence = max(0.0, min(1.0, confidence))

                # Determine potential fraud types based on score thresholds
                fraud_types = []
                
                # Add logic to identify potential fraud types based on prediction scores
                # For example:
                if ensemble_score > 0.8:
                    fraud_types.append('HIGH_RISK_TRANSACTION')
                elif ensemble_score > 0.6:
                    fraud_types.append('SUSPICIOUS_ACTIVITY')
                
                # You could add more sophisticated fraud type detection here
                # For example, if you have specific model outputs that indicate different fraud types
                
                return {
                    'score': float(ensemble_score),
                    'confidence': float(confidence),
                    'fraud_types': fraud_types
                }
            
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {str(e)}")
            return {
                'score': 0.0,
                'confidence': 0.0,
                'fraud_types': []
            }
    
class ModelTrainer:
    """Handles training and evaluation of the ensemble model"""
    
    def __init__(self, feature_extractor: TransactionFeatureExtractor):
        self.feature_extractor = feature_extractor
        
    
    
    def train_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, MLModel]:
        """Train all models in the ensemble"""
        models = {
            'random_forest': RandomForestModel(),
            'neural_net': NeuralNetModel(),
            'xgboost': XGBoostModel()
        }
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            model.fit(X, y)
            
        return models
    
    def evaluate_models(self, models: Dict[str, MLModel], X: np.ndarray, y: np.ndarray) -> Dict[str, Dict]:
        """Evaluate each model's performance"""
        results = {}
        for name, model in models.items():
            y_pred = model.model.predict(X)
            y_prob = model.model.predict_proba(X)[:, 1]
            
            results[name] = {
                'precision': precision_score(y, y_pred),
                'recall': recall_score(y, y_pred),
                'f1': f1_score(y, y_pred),
                'auc_roc': roc_auc_score(y, y_prob)
            }
            
        return results



class NAICSRiskChecker:
    def __init__(self, config_reader: ConfigReader):
        """
        Initialize NAICS Risk Checker
        
        :param config_reader: Configuration reader
        """
        self.config_reader = config_reader
        self.naics_risk_configs = self._load_naics_risk_configs()
    
    def _load_naics_risk_configs(self) -> List[Dict]:
        """
        Load NAICS risk configurations from properties
        
        :return: List of NAICS risk configuration dictionaries
        """
        naics_configs = []
        global_used_codes = set()  # Track used codes globally
        
        # Read up to 10 possible NAICS configurations
        for i in range(1, 11):
            try:
                # Construct config keys
                codes_key = f'naics.risk.config.{i}.codes'
                score_key = f'naics.risk.config.{i}.default_base_score'
                
                # Try to read configuration
                codes = self.config_reader.get_property(codes_key, '').strip()
                base_score = float(self.config_reader.get_property(score_key, '0'))

                 # Skip configurations with no codes
                if not codes:
                    # logger.info(f"Skipping NAICS config {i} due to empty codes")
                    continue  # No more configurations

                # Clean and validate codes
                config_codes = set(code.strip() for code in codes.split(','))
                
                # Check for duplicate codes
                duplicate_codes = config_codes.intersection(global_used_codes)
                if duplicate_codes:
                    #logger.warning(f"Duplicate NAICS codes found in config {i}: {duplicate_codes}")
                    # Remove duplicate codes
                    config_codes -= duplicate_codes
                
                # Skip if no unique codes remain
                if not config_codes:
                    #logger.warning(f"No unique NAICS codes in config {i}")
                    continue
                
                # Add unique codes to global set
                global_used_codes.update(config_codes)
                
                naics_configs.append({
                    'codes': config_codes,
                    'default_base_score': base_score
                })

            
            except Exception as e:
                logger.error(f"Error loading NAICS config {i}: {str(e)}")

        # Sort configurations by default base score in descending order
        naics_configs.sort(key=lambda x: x['default_base_score'], reverse=True)
        
        return naics_configs
    
    def check_naics_risk(self, naics_codes: List[str]) -> Dict:
        """
        Check risk for given NAICS codes
        
        :param naics_codes: List of NAICS codes to check
        :return: Risk assessment dictionary
        """
        # Convert input to set for efficient lookup
        naics_set = set(naics_codes)
        
        # Find matching configurations
        matching_configs = []
        matched_codes = set()

        for config in self.naics_risk_configs:
            # Find codes in this configuration that haven't been matched before
            new_matched_codes = config['codes'].intersection(naics_set - matched_codes)
        
            if new_matched_codes:
                matching_configs.append({
                    **config,
                    'matched_codes': new_matched_codes
                })
                matched_codes.update(new_matched_codes)
    
        # If no matches, return default low risk
        if not matching_configs:
            return {
                'risk_score' : 0,
                'matched_naics_codes': []
                
            }
        
        # Take the highest risk configuration (already sorted)
        highest_risk_config = matching_configs[0]

        # Prepare risk assessment
        return {
            'risk_score': highest_risk_config['default_base_score'],
            'matched_naics_codes': [
                code for config in matching_configs 
                for code in config['codes'] 
                if code in naics_set
            ]
        }

class FraudDetectionSystem:
    """Main system class combining parsing and detection"""
    
    def __init__(self, config_path: str = "./config.properties"):
        # print(f"DEBUG: FraudDetectionSystem.__init__ - config_path='{config_path}'")

        self.logger = logging.getLogger(__name__)


        try:
            # Load configuration
            self.config_reader = ConfigReader(config_path)
            
            # Get logger
            self.logger = logging.getLogger(__name__)
            self.logger.info("FraudDetectionSystem initialization started")
            
            # Database connection initialization
            try:
                db_type = self.config_reader.get_property('database.type', 'sql')
                self.logger.info(f"Database initialization starting with type: {db_type}")
                
                # Add detailed debug for each step of database initialization
                if db_type.lower() == 'sql':
                    self.logger.info("Initializing SQL database connection")
                    host = self.config_reader.get_property('db.host', '')
                    user = self.config_reader.get_property('db.user', '')
                    db_name = self.config_reader.get_property('db.database', '')
                    password = self.config_reader.get_property('db.password', '')
                    port = self.config_reader.get_property('db.port', '3306')
                    
                    self.logger.info(f"Creating database connection with: host={host}, user={user}, db={db_name}, port={port}")
                    
                    # Try step by step
                    import mysql.connector
                    self.logger.info("MySQL connector imported successfully")
                    
                    # Step 1: Create config dictionary
                    db_config = {
                        'host': host,
                        'user': user,
                        'password': password,
                        'database': db_name,
                        'port': int(port)
                    }
                    
                    # Check for SSL settings
                    ssl_enabled = self.config_reader.get_property('database.ssl', 'false').lower() == 'true'
                    if ssl_enabled:
                        self.logger.info("SSL is enabled for database connection")
                        ssl_ca = self.config_reader.get_property('database.ssl.ca', None)
                        if ssl_ca:
                            db_config['ssl_ca'] = ssl_ca
                        else:
                            db_config['ssl_disabled'] = False
                    
                    self.logger.info("Attempting to create SQL database connection")
                    
                    # Step 2: Create connection (with timeout)
                    try:
                        self.db_connection = mysql.connector.connect(**db_config)
                        self.logger.info("SQL database connection successful")
                    except mysql.connector.Error as e:
                        self.logger.error(f"MySQL connection error: {e}")
                        raise

            except Exception as e:
                self.logger.error(f"Database initialization failed: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                raise

        
        except Exception as e:
            self.logger.error(f"FraudDetectionSystem initialization failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise  # Re-raise to ensure the error is visible higher up

        # Initialize config reader first
        self.config_reader = ConfigReader(config_path)
        # print(f"DEBUG: ConfigReader initialized: {type(self.config_reader)}")
        self.config_reader.load_properties()
        # print(f"DEBUG: Properties loaded into config_reader")
    
        # Initialize risk thresholds right at the start
    
        self.ACH_RETURN_CODES = {}
        
        for code in ['R01', 'R02', 'R03', 'R04', 'R05', 'R06', 'R07', 'R08', 'R09', 'R10', 
                    'R11', 'R12', 'R14', 'R15', 'R16', 'R17', 'R20', 'R29']:
            category = self.config_reader.get_property(f"ach.return.{code}.category", "ADMINISTRATIVE")
            severity = self.config_reader.get_property(f"ach.return.{code}.severity", "MEDIUM")
            
            # Get description from config or use a default
            description = self.config_reader.get_property(f"ach.return.{code}.description", f"Return Code {code}")
            
            self.ACH_RETURN_CODES[code] = {
                'text': description,
                'category': category,
                'severity': severity
            }
        self.risk_thresholds = {
            'velocity_24h': int(self.config_reader.get_property('risk.threshold.velocity_24h', '5')),
            'amount_24h': float(self.config_reader.get_property('risk.threshold.amount_24h', '10000')),
            'unique_recipients_24h': int(self.config_reader.get_property('risk.threshold.unique_recipients_24h', '3')),
            'amount_deviation': float(self.config_reader.get_property('risk.threshold.amount_deviation', '2.0')),
            'fraud_score': int(self.config_reader.get_property('risk.threshold.fraud_score', '60')),
            'ml_score': float(self.config_reader.get_property('risk.threshold.ml_score', '0.7')),
            'pattern_score': float(self.config_reader.get_property('risk.threshold.pattern_score', '0.6'))
        }

        self.parser = ISO20022Parser()
        self.timezone_manager = TimeZoneManager()
        
        # Get database type from configuration
        db_type = self.config_reader.get_property('database.type', 'sql')
        logger.info(f"Database type from config in main(): '{db_type}'")

        # Initialize appropriate database connection based on configuration
        if db_type == 'csv':
            self.db = CSVDatabaseConnection(config_path)
        else:  # Default to SQL database
            self.db = DatabaseConnection(config_path)

        # Load ACH return codes from configuration
    

        # Add channel detector initialization
        self.channel_detector = ChannelAlertDetector(self.config_reader)
        self.analyzer = TransactionAnalyzer()
        # print(f"DEBUG: About to create TransactionFeatureExtractor with config_reader={type(self.config_reader)}")
        self.feature_extractor = TransactionFeatureExtractor(config_reader=self.config_reader,channel_detector=self.channel_detector)
        # print(f"DEBUG: Explicitly setting feature_extractor.config_reader")
        self.feature_extractor.config_reader = self.config_reader  # Explicit assignment to ensure it's set
        # print(f"DEBUG: TransactionFeatureExtractor created")
        self.decision_controller = DecisionController(self.config_reader)
        self.mule_detector = MuleAccountDetector(self.config_reader, self.db)
        self.ato_detector = AccountTakeoverDetector(self.config_reader)
        # Add APP Fraud Pattern Detector
        self.app_detector = APPFraudDetector(self.config_reader)

        # Set timezone manager on all components that need it
        self.channel_detector.timezone_manager = self.timezone_manager
        self.app_detector.timezone_manager = self.timezone_manager
        self.feature_extractor.timezone_manager = self.timezone_manager
        self.mule_detector.timezone_manager = self.timezone_manager
        self.ato_detector.timezone_manager = self.timezone_manager

        # Initialize model weights
        self.model_weights = {
            'random_forest': 0.3,
            'neural_net': 0.3,
            'xgboost': 0.4
        }

        # Create sample training data for feature extractor and models
        try:
            sample_transactions = [
                {
                    'amount': 1000.0,
                    'channel': 'WIRE',
                    'currency': 'USD',
                    'debtor_country': 'US',
                    'creditor_country': 'GB',
                    'is_cross_border': True,
                    'involves_high_risk_country': False,
                    'velocity_24h': 1,
                    'amount_24h': 1000.0,
                    'unique_recipients_24h': 1,
                    'velocity_7d': 3,
                    'amount_7d': 3000.0,
                    'unique_recipients_7d': 2,
                    'avg_amount_30d': 800.0,
                    'std_amount_30d': 200.0
                },
                {
                    'amount': 500.0,
                    'channel': 'ACH',
                    'currency': 'USD',
                    'debtor_country': 'US',
                    'creditor_country': 'US',
                    'is_cross_border': False,
                    'involves_high_risk_country': False,
                    'velocity_24h': 2,
                    'amount_24h': 1500.0,
                    'unique_recipients_24h': 2,
                    'velocity_7d': 5,
                    'amount_7d': 2500.0,
                    'unique_recipients_7d': 3,
                    'avg_amount_30d': 600.0,
                    'std_amount_30d': 150.0
                },
                {
                    'amount': 5000.0,
                    'channel': 'WIRE',
                    'currency': 'USD',
                    'debtor_country': 'US',
                    'creditor_country': 'IR',
                    'is_cross_border': True,
                    'involves_high_risk_country': True,
                    'velocity_24h': 3,
                    'amount_24h': 8000.0,
                    'unique_recipients_24h': 3,
                    'velocity_7d': 8,
                    'amount_7d': 15000.0,
                    'unique_recipients_7d': 5,
                    'avg_amount_30d': 2000.0,
                    'std_amount_30d': 1000.0
                }
            ]

            # First fit the feature extractor
            self.feature_extractor.fit(sample_transactions)

            # Initialize models
            self.models = {
                'random_forest': RandomForestModel(),
                'neural_net': NeuralNetModel(),
                'xgboost': XGBoostModel()
            }

            # Process each transaction one by one
            features_list = []
            for tx in sample_transactions:
                # Create metrics object
                metrics = TransactionMetrics(
                    velocity_24h=tx['velocity_24h'],
                    amount_24h=tx['amount_24h'],
                    unique_recipients_24h=tx['unique_recipients_24h'],
                    velocity_7d=tx['velocity_7d'],
                    amount_7d=tx['amount_7d'],
                    unique_recipients_7d=tx['unique_recipients_7d'],
                    avg_amount_30d=tx['avg_amount_30d'],
                    std_amount_30d=tx['std_amount_30d'],
                    new_recipient=True,
                    cross_border=tx['is_cross_border'],
                    high_risk_country=tx['involves_high_risk_country']
                )

                # Extract features
                try:
                    features = self.feature_extractor.extract_features(tx, metrics)
                    if isinstance(features, np.ndarray):
                        if len(features.shape) == 1:
                            features = features.reshape(1, -1)
                        features_list.append(features)
                    else:
                        logger.error(f"Unexpected feature type: {type(features)}")
                        continue
                except Exception as e:
                    logger.error(f"Error extracting features: {str(e)}")
                    continue

            # Stack features and create labels
            if features_list:
                X = np.vstack(features_list)
                y = np.array([0, 0, 1])  # Labels for our three samples

                # Verify shapes
                # logger.info(f"Training data shapes - X: {X.shape}, y: {y.shape}")
                
                if X.shape[0] == len(y):
                    # Fit each model
                    for name, model in self.models.items():
                        try:
                            # logger.info(f"Fitting {name} model...")
                            model.fit(X, y)
                            # logger.info(f"{name} model fitted successfully")
                        except Exception as e:
                            logger.error(f"Error fitting {name} model: {str(e)}")
                else:
                    logger.error(f"Shape mismatch: X has {X.shape[0]} samples but y has {len(y)} samples")
            else:
                logger.error("No valid features extracted from sample transactions")

        except Exception as e:
            logger.error(f"Error during model initialization: {str(e)}")
                
        # Initialize detector after analyzer and feature extractor
        self.detector = EnsembleFraudDetectionModel(
                config_reader=self.config_reader,
                db_connection=self.db
            )
        
        self.folders = {
            'incoming': Path('incoming_messages'),
            'processed': Path('processed_messages'),
            'failed': Path('failed_messages')
        }
        
        # # Add OFAC SDN Checker
        self.ofac_checker = OFACSDNChecker(self.config_reader)
        
        # Initialize PlatformListManager
        self._initialize_platform_list_manager()
        if hasattr(self, 'platform_list_manager') and self.platform_list_manager is not None:
            self.logger.info("PlatformListManager is available after initialization")
        else:
            self.logger.error("PlatformListManager initialization failed - attribute not set")
        # Add NAICS Risk Checker
        self.naics_risk_checker = NAICSRiskChecker(self.config_reader)
        

        # Check if we should load pre-trained models
        use_trained_models = self.config_reader.get_property('models.use_trained', 'false') == 'true'
        model_timestamp = self.config_reader.get_property('models.timestamp', '')

        # After initializing everything else, try to load pre-trained models if requested
        if use_trained_models and model_timestamp:
            model_dir = self.config_reader.get_property('models.directory', './trained_models')
            # This is the change - using the standalone function instead of self.load_models
            success, message = load_trained_models(self, model_timestamp, model_dir)
            if success:
                # Ensure attributes are set
                self.feature_extractor.config_reader = self.config_reader
                self.feature_extractor.channel_detector = self.channel_detector            
                logger.info(f"Successfully loaded pre-trained models with timestamp {model_timestamp}")
            else:
                logger.warning(f"Failed to load pre-trained models: {message}")
                logger.info("Using default/sample-trained models instead")

        from business_explainer import BusinessExplainer
        self.business_explainer = BusinessExplainer(self.config_reader)

        from transaction_logger import TransactionLogger

        # Initialize transaction logger
        transaction_log_dir = self.config_reader.get_property('transaction.log.directory', './transaction_logs')
        self.transaction_logger = TransactionLogger(self.config_reader, transaction_log_dir)
        logger.info("Transaction logger initialized")

    def _initialize_platform_list_manager(self):
        """Initialize the Platform List Manager with configuration from config_reader"""
        try:
            
            # Get Redis configuration from config_reader
            redis_config = {
                "host": self.config_reader.get_property('redis.host', 'localhost'),
                "port": int(self.config_reader.get_property('redis.port', '6379')),
                "db": int(self.config_reader.get_property('redis.db', '0'))
            }
            
            # Create the PlatformListManager instance
            self.platform_list_manager = PlatformListManager(self.db, redis_config)
            self.logger.info("PlatformListManager initialized successfully")
        
        except Exception as e:
            self.logger.error(f"Failed to initialize PlatformListManager: {str(e)}")
            # Set to None so we can check if it's available in methods that use it
            self.platform_list_manager = None
    
    def refresh_platform_lists(self):
        """Refresh all platform lists from the database"""
        try:
            if hasattr(self, 'platform_list_manager') and self.platform_list_manager:
                self.platform_list_manager.refresh_from_database()
                self.logger.info("Platform lists refreshed successfully")
            else:
                self.logger.warning("Cannot refresh platform lists - PlatformListManager not initialized")
        except Exception as e:
            self.logger.error(f"Error refreshing platform lists: {str(e)}")

    def train_models_from_csv(self, csv_path: str, save_path: Optional[str] = None) -> Dict[str, Dict]:
        """
        Train models using a CSV dataset and optionally save them
        
        :param csv_path: Path to the CSV training data
        :param save_path: Directory to save trained models (optional)
        :return: Dictionary with model evaluation metrics
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Loading training data from {csv_path}")
        
        try:
            # Load the CSV data
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(df)} training samples")
            
            # Prepare data for training
            transactions = self._prepare_transactions_from_df(df)
            metrics_list = self._prepare_metrics_from_df(df)
            labels = df['is_fraudulent'].values
            
            # Initialize or reset feature extractor
            self.feature_extractor.fit(transactions)
            
            # Extract features
            features_list = []
            valid_indices = []
            
            for i, (transaction, metrics) in enumerate(zip(transactions, metrics_list)):
                try:
                    features = self.feature_extractor.extract_features(transaction, metrics)
                    features_list.append(features)
                    valid_indices.append(i)
                except Exception as e:
                    logger.error(f"Error extracting features for transaction {i}: {str(e)}")
                    continue
            
            # Stack features and get corresponding labels
            X = np.vstack(features_list)
            y = np.array([labels[i] for i in valid_indices])
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train models
            for name, model in self.models.items():
                logger.info(f"Training {name} model")
                model.fit(X_train, y_train)
            
            # Evaluate models
            evaluation_results = {}
            for name, model in self.models.items():
                # Get predictions
                y_pred = model.model.predict(X_val)
                y_prob = model.model.predict_proba(X_val)[:, 1]
                
                # Calculate metrics
                evaluation_results[name] = {
                    'precision': float(precision_score(y_val, y_pred)),
                    'recall': float(recall_score(y_val, y_pred)),
                    'f1': float(f1_score(y_val, y_pred)),
                    'auc_roc': float(roc_auc_score(y_val, y_prob))
                }
                
                logger.info(f"Model {name} metrics: {evaluation_results[name]}")
            
            # Save models if requested
            if save_path:
                os.makedirs(save_path, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Save model evaluation results
                with open(os.path.join(save_path, f"model_metrics_{timestamp}.json"), 'w') as f:
                    json.dump(evaluation_results, f, indent=2)

                # Save feature extractor
                feature_extractor_path = os.path.join(save_path, f"feature_extractor_{timestamp}.pkl")
                with open(feature_extractor_path, 'wb') as f:
                    pickle.dump(self.feature_extractor, f)
                logger.info(f"Feature extractor saved to {feature_extractor_path}")

                # Save each model
                for name, model in self.models.items():
                    model_path = os.path.join(save_path, f"{name}_model_{timestamp}.pkl")
                    with open(model_path, 'wb') as f:
                        pickle.dump(model, f)
                    logger.info(f"Model {name} saved to {model_path}")
                    
                logger.info(f"All models and model metrics saved to {save_path} with timestamp {timestamp}")
                
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Error training models from CSV: {str(e)}")
            return {}

    def _prepare_transactions_from_df(self, df: pd.DataFrame) -> List[Dict]:
        """
        Convert dataframe rows to transaction dictionaries
        
        :param df: DataFrame with transaction data
        :return: List of transaction dictionaries
        """
        transactions = []
        
        for idx, row in df.iterrows():
            # Convert DbtrAccountOpenDate and TransactionDate to datetime if they're strings
            dbtr_account_open_date = pd.to_datetime(row['DbtrAccountOpenDate']) if 'DbtrAccountOpenDate' in row else None
            transaction_date = pd.to_datetime(row['TransactionDate']) if 'TransactionDate' in row else datetime.now()
            
            transaction = {
                'msg_id': f"TRAIN_{idx}",
                'pmt_info_id': row.get('Pmt_Inf_Id', f"TRAIN_PMT_{idx}"),
                'amount': float(row['amount']),
                'channel': str(row['channel']),
                'currency': str(row.get('currency', 'USD')),
                'debtor_country': str(row['debtor_country']),
                'creditor_country': str(row.get('creditor_country', 'US')),
                'creditor_name': str(row.get('creditor_name', f"Training_Creditor_{idx}")),
                'is_cross_border': bool(row['is_cross_border']),
                'involves_high_risk_country': bool(row['involves_high_risk_country']),
                'creation_date': transaction_date,
                'account_creation_date': dbtr_account_open_date,
                'debtor_name': str(row.get('debtor_name', f"Training_Debtor_{idx}")),
                'debtor_account': str(row.get('DbtrAccountNumber', f"ACC_{idx}_DEBTOR")),
                'creditor_account': str(row.get('CrdtrAccountNumber', f"ACC_{idx}_CREDITOR")),
                'debtor_account_routing_number': str(row.get('DbtrAccountRoutingNumber', '')),
                'creditor_account_routing_number': str(row.get('CrdtrAccountRoutingNumber', '')),
                'required_execution_date': str(row.get('ReqdExctnDt', '')),
                'days_to_execution': int(row.get('days_to_execution', 0)),
                'account_age_days': int(row.get('account_age_days', 0)),
                'hour_of_day': int(row.get('hour_of_day', 12)),
                'day_of_week': int(row.get('day_of_week', 0)),
                'weekday': bool(row.get('weekday', True)),
                'is_business_hours': bool(row.get('is_business_hours', True)),
                'user_id': str(row.get('Dbtr_User_Id', '')),
                'is_round_amount': bool(row.get('is_round_amount', False)),
                'is_weekend': True if row.get('day_of_week', 0) >= 5 else False,
                'local_hour': int(row.get('hour_of_day', 12)),
                'is_high_risk_hour': not bool(row.get('is_business_hours', True))
            }
            transactions.append(transaction)
        
        return transactions

    def _prepare_metrics_from_df(self, df: pd.DataFrame) -> List[TransactionMetrics]:
        """
        Create TransactionMetrics objects from the dataframe with expanded metrics
        
        :param df: DataFrame with transaction metrics
        :return: List of TransactionMetrics objects
        """
        metrics_list = []
        
        for _, row in df.iterrows():
            # Create TransactionMetrics with all available fields from CSV
            metrics = TransactionMetrics(
                # Basic metrics (always include these)
                velocity_24h=int(row.get('velocity_24h', 0)),
                amount_24h=float(row.get('amount_24h', 0)),
                unique_recipients_24h=int(row.get('unique_recipients_24h', 0)),
                velocity_7d=int(row.get('velocity_7d', 0)),
                amount_7d=float(row.get('amount_7d', 0)),
                unique_recipients_7d=int(row.get('unique_recipients_7d', 0)),
                avg_amount_30d=float(row.get('avg_amount_30d', float(row['amount']))),
                std_amount_30d=float(row.get('amount_30d_std', 0)),
                new_recipient=bool(row.get('new_recipient', False)),
                cross_border=bool(row.get('is_cross_border', False)),
                high_risk_country=bool(row.get('involves_high_risk_country', False)),
                
                # Account-level metrics
                account_age_days=int(row.get('account_age_days', 0)),
                activity_days_before_transaction=int(row.get('activity_days_before_transaction', 0)),
                days_since_first_transaction=int(row.get('days_since_first_transaction', 0)),
                days_since_last_transaction=int(row.get('days_since_last_transaction', 0)),
                
                # Time-related metrics
                weekday=int(row.get('day_of_week', 0)),
                hour_of_day=int(row.get('hour_of_day', 12)),
                is_business_hours=bool(row.get('is_business_hours', True)),
                typical_transaction_hour_flag=bool(row.get('typical_transaction_hour_flag', True)),
                typical_transaction_weekday_flag=bool(row.get('typical_transaction_weekday_flag', True)),
                unusual_access_time_flag=bool(row.get('unusual_access_time_flag', False)),
                
                # Pattern metrics
                amount_zscore=float(row.get('amount_zscore', 0)),
                rounded_amount_ratio=float(row.get('rounded_amount_ratio', 0)),
                funnel_pattern_ratio=float(row.get('funnel_pattern_ratio', 0)),
                off_hours_transaction_ratio=float(row.get('off_hours_transaction_ratio', 0)),
                rapid_withdrawal_flag=bool(row.get('rapid_withdrawal_flag', False)),
                account_emptying_percentage=float(row.get('account_emptying_percentage', 0)),
                velocity_ratio=float(row.get('velocity_ratio', 1.0)),
                
                # Risk scores
                transaction_pattern_break_risk_score=float(row.get('transaction_pattern_break_risk_score', 0)),
                amount_anomaly_risk_score=float(row.get('amount_anomaly_risk_score', 0)),
                new_recipient_risk_score=float(row.get('new_recipient_risk_score', 0)),
                round_amount_risk_score=float(row.get('round_amount_risk_score', 0)),
                unusual_timing_risk_score=float(row.get('unusual_timing_risk_score', 0)),
                after_hours_risk_score=float(row.get('after_hours_risk_score', 0)),
                unusual_location_risk_score=float(row.get('unusual_location_risk_score', 0)),
                velocity_risk_score=float(row.get('velocity_risk_score', 0)),
                
                # Boolean flags
                is_round_amount=bool(row.get('is_round_amount', False)),
                app_fraud_risk=float(row.get('app_fraud_risk', 0)),
                known_scam_recipient_flag=bool(row.get('known_scam_recipient_flag', False)),
                unusual_location_flag=bool(row.get('unusual_location_flag', False)),
                geo_velocity_impossible_flag=bool(row.get('geo_velocity_impossible_flag', False)),
                vpn_proxy_flag=bool(row.get('vpn_proxy_flag', False)),
                
                # Additional metrics requested
                behavioral_consistency_score=float(row.get('behavioral_consistency_score', 0)),
                transaction_predictability_score=float(row.get('transaction_predictability_score', 0)),
                beneficiary_risk_score=float(row.get('beneficiary_risk_score', 0)),
                new_beneficiary_flag=bool(row.get('new_beneficiary_flag', False)),
                new_beneficiary_risk_score=float(row.get('new_beneficiary_risk_score', 0)),
                amount_pattern_risk_score=float(row.get('amount_pattern_risk_score', 0)),
                unusual_time_risk_score=float(row.get('unusual_time_risk_score', 0)),
                multiple_beneficiaries_risk_score=float(row.get('multiple_beneficiaries_risk_score', 0)),
                triggered_indicators_count=int(row.get('triggered_indicators_count', 0)),
                is_suspected_app_fraud=bool(row.get('is_suspected_app_fraud', False)),
                is_suspected_takeover=bool(row.get('is_suspected_takeover', False)),
                is_account_takeover=bool(row.get('is_account_takeover', False)),
                multi_indicator_present=bool(row.get('multi_indicator_present', False)),
                
                # Additional pattern and trend metrics
                account_milestone_deviation=float(row.get('account_milestone_deviation', 0)),
                amount_volatility_365d=float(row.get('amount_volatility_365d', 0)),
                annual_amount_trend=float(row.get('annual_amount_trend', 0)),
                annual_velocity_percentile=float(row.get('annual_velocity_percentile', 0)),
                behavioral_change_velocity=float(row.get('behavioral_change_velocity', 0)),
                confidence_level=str(row.get('confidence_level', "LOW")),
                payment_growth_rate=float(row.get('payment_growth_rate', 0)),
                pattern_break_frequency_365d=int(row.get('pattern_break_frequency_365d', 0)),
                quarterly_activity_comparison=float(row.get('quarterly_activity_comparison', 0)),
                recipient_concentration_ratio=float(row.get('recipient_concentration_ratio', 0)),
                recipient_network_density=float(row.get('recipient_network_density', 0)),
                risk_volatility_365d=float(row.get('risk_volatility_365d', 0)),
                rolling_risk_trend_365d=float(row.get('rolling_risk_trend_365d', 0)),
                seasonal_adjustment_factor=float(row.get('seasonal_adjustment_factor', 1.0)),
                seasonality_pattern_deviation=float(row.get('seasonality_pattern_deviation', 0)),
                year_over_year_behavior_change=float(row.get('year_over_year_behavior_change', 0)),
                
                # Authentication and device metrics
                credential_age_days=int(row.get('credential_age_days', 999)),
                device_change_recency=int(row.get('device_change_recency', 999)),
                failed_login_attempts_24h=int(row.get('failed_login_attempts_24h', 0)),
                historical_fraud_attempts=int(row.get('historical_fraud_attempts', 0)),
                login_frequency_30d=float(row.get('login_frequency_30d', 0)),
                mfa_bypass_attempts=int(row.get('mfa_bypass_attempts', 0)),
                password_reset_recency=int(row.get('password_reset_recency', 999)),
                session_duration=int(row.get('session_duration', 0)),
                
                # Recipient metrics
                days_since_first_payment_to_recipient=int(row.get('days_since_first_payment_to_recipient', 999)),
                recipient_inactivity_period=int(row.get('recipient_inactivity_period', 0)),
                recipient_transaction_history_365d=row.get('recipient_transaction_history_365d', []),
                source_target_previous_transfers=int(row.get('source_target_previous_transfers', 0)),
                threshold_days_since_last_recipient=int(row.get('threshold_days_since_last_recipient', 999)),
                total_unique_recipients_365d=int(row.get('total_unique_recipients_365d', 0)),
                
                # Transaction volume metrics
                transaction_frequency_stddev=float(row.get('transaction_frequency_stddev', 0)),
                amount_30d=float(row.get('amount_30d', 0)),
                avg_amount_24h=float(row.get('avg_amount_24h', 0)),
                avg_amount_7d=float(row.get('avg_amount_7d', 0)),
                amount_30d_avg=float(row.get('amount_30d_avg', 0)),
                amount_30d_std=float(row.get('amount_30d_std', 0)),
                avg_monthly_transaction_volume=float(row.get('avg_monthly_transaction_volume', 0)),
                tx_count_1h=float(row.get('tx_count_1h', 0)),
                route_count_1h=float(row.get('route_count_1h', 0)),
                avg_monthly_transaction_count=float(row.get('avg_monthly_transaction_count', 0)),
                
                # Handle recipient_frequency_distribution (this is likely a dictionary)
                recipient_frequency_distribution=row.get('recipient_frequency_distribution', {}),
                
                # Additional boolean flags from your list
                new_recipient_large_amount=bool(row.get('new_recipient_large_amount', False)),
                periodic_payment_disruption=bool(row.get('periodic_payment_disruption', False)),
                recipient_pattern_change_flag=bool(row.get('recipient_pattern_change_flag', False)),
            )
            metrics_list.append(metrics)
        
        return metrics_list
    
    def load_models(system, timestamp, model_dir="./trained_models"):
        """
        Load trained models and feature extractor from disk
        
        :param timestamp: Timestamp string used when models were saved
        :param model_dir: Directory containing the saved models
        :return: Tuple of (success_flag, error_message)
        """
        try:
            logger.info(f"Loading models with timestamp {timestamp} from {model_dir}")
        
            # Load feature extractor
            extractor_path = f"{model_dir}/feature_extractor_{timestamp}.pkl"
            if not os.path.exists(extractor_path):
                return False, f"Feature extractor file not found: {extractor_path}"
                    
            with open(extractor_path, 'rb') as f:
                system.feature_extractor = pickle.load(f)
            
            # Add this line if it's missing
            system.feature_extractor.config_reader = system.config_reader
            
            # Add these explicit INFO logs to verify the assignment
            logger.info(f"CONFIG CHECK: feature_extractor.config_reader={type(system.feature_extractor.config_reader)}")
            logger.info(f"CONFIG CHECK: system.config_reader={type(system.config_reader)}")
            
            logger.info("Feature extractor loaded successfully")
            
            logger.info("Feature extractor loaded successfully")
            
            
            # Load models
            model_names = ['random_forest', 'neural_net', 'xgboost']
            loaded_models = {}
            
            for name in model_names:
                model_path = f"{model_dir}/{name}_model_{timestamp}.pkl"
                if not os.path.exists(model_path):
                    return False, f"Model file not found: {model_path}"
                    
                with open(model_path, 'rb') as f:
                    loaded_models[name] = pickle.load(f)
                logger.info(f"Model {name} loaded successfully")
            
            # Update instance models
            system.models = loaded_models
            
            # Verify model loading by checking if models have required attributes
            for name, model in system.models.items():
                if not hasattr(model, 'model') or not hasattr(model.model, 'n_features_in_'):
                    return False, f"Model {name} appears to be invalid or untrained"
            
            logger.info("All models loaded and verified successfully")
            return True, "Models loaded successfully"
            
        except Exception as e:
            error_message = f"Error loading models: {str(e)}"
            logger.error(error_message)
            return False, error_message

    def _initialize_result(self) -> Dict:
        """Initialize result dictionary with default values"""
        return {
            'transaction_id': 'UNKNOWN',
            'final_risk_score': 0,
            'alerts': [],
            'metrics': {},
            'risk_factors': {
                'compliance_alerts': [],
                'velocity_alerts': [],
                'amount_alerts': [],
                'geographic_alerts': [],
                'recipient_alerts': [],
                'pattern_alerts': [],
                'channel_alerts': [],
                'triggered_rules': [],
                'score_components': {
                    'base_score': 0,
                    'high_risk_penalty': 0,
                    'ml_score': 0,
                    'consolidated_score': 0,
                    'channel_score': 0,
                    'final_score': 0
                }
            },
            'model_scores': {},
            'fraud_types': [],
            'risk_level': 'LOW',
            'bypass_reason': None,
            'whitelist_match': None,
            'timestamp': datetime.now().isoformat(),
            'alert_generated':False,
            'channel': 'UNKNOWN',
            'ensemble_score': 0,
            'channel_score': 0,
            'app_fraud_score': 0,
            'app_fraud_patterns': [],
            'app_behavioral_flags': []
        }


    def _extract_entity_id(self, content: str) -> int:
        """Extract entity ID from message content with better error handling"""
        try:
            if not content:
                #logger.warning("Empty message content")
                return 1

            root = ET.fromstring(content)
            
            # Try different namespace patterns
            namespace_patterns = [
                # PAIN.001
                ('.//pain:InitgPty/pain:Id/pain:OrgId/pain:Othr/pain:Id', 
                {'pain': 'urn:iso:std:iso:20022:tech:xsd:pain.001.001.09'}),
                # PACS.008
                ('.//pacs:InitgPty/pacs:Id/pacs:OrgId/pacs:Othr/pacs:Id',
                {'pacs': 'urn:iso:std:iso:20022:tech:xsd:pacs.008.001.08'}),
                # CAMT.054
                ('.//camt:InitgPty/camt:Id/camt:OrgId/camt:Othr/camt:Id',
                {'camt': 'urn:iso:std:iso:20022:tech:xsd:camt.054.001.08'}),
                # Alternative patterns
                ('.//Othr/Id', {}),
                ('.//OrgId/Othr/Id', {}),
                ('.//InitgPty//Id', {})
            ]
            
            # Try each pattern
            for xpath, ns in namespace_patterns:
                try:
                    elements = root.findall(xpath, ns)
                    for element in elements:
                        if element is not None and element.text:
                            try:
                                # Try to convert to integer
                                return int(element.text)
                            except ValueError:
                                # If conversion fails, try to extract numbers
                                numbers = re.findall(r'\d+', element.text)
                                if numbers:
                                    return int(numbers[0])
                                continue
                except Exception:
                    continue
            
            # logger.warning("No valid entity ID found in message")
            return 1
            
        except ET.ParseError:
            logger.error("Invalid XML content")
            return 1
        except Exception as e:
            logger.error(f"Error extracting entity ID: {str(e)}")
            return 1
    
    def get_ensemble_prediction(self, features: np.ndarray) -> float:
        """Get ensemble prediction from detector"""
        try:
            # Delegate to detector's get_ensemble_prediction method
            return self.detector.get_ensemble_prediction(features)
        except Exception as e:
            logger.error(f"Error getting ensemble prediction: {str(e)}")
            return 0.0
    
    def _perform_business_entity_checks(self, transaction: Dict) -> Dict:
        """Perform business entity and NAICS checks with error handling"""
        default_result = {
            'is_business': False,
            'risk_score': 0,
            'debtor_business_details': {'is_business': False},
            'creditor_business_details': {'is_business': False},
            'naics_risk': {
                'risk_score': 0,
                'matched_naics_codes': []
            }
        }

        try:
            # Check debtor name
            debtor_check = self.db.check_business_entity(
                transaction.get('debtor_name', '')
            ) or {'is_business': False}
            
            # Check creditor name
            creditor_check = self.db.check_business_entity(
                transaction.get('creditor_name', '')
            ) or {'is_business': False}
            
            # Determine business entity status
            is_business = debtor_check.get('is_business', False) or creditor_check.get('is_business', False)
            
            # Collect NAICS codes
            naics_codes = (
                debtor_check.get('naics_codes', []) + 
                creditor_check.get('naics_codes', [])
            )
            
            # Check NAICS risk
            naics_risk = self.naics_risk_checker.check_naics_risk(naics_codes)
            
            return {
                'is_business': is_business,
                'debtor_business_details': debtor_check,
                'creditor_business_details': creditor_check,
                'risk_score': naics_risk.get('risk_score', 0),
                'naics_risk': naics_risk
            }
        
        except Exception as e:
            logger.error(f"Error performing business entity checks: {str(e)}")
            return default_result
        
    def _check_ach_return_rates(self, transaction: Dict) -> Dict:
        """
        Perform ACH return rate monitoring and checks
        
        Args:
            transaction (Dict): Transaction details
        
        Returns:
            Dict: Return rate check results with alerts and risk assessment
        """
        #Data is not available hence always return false..remember to remove this return later

        
        # Only perform for ACH channel
        if transaction.get('channel') != 'ACH':
            return {
                'is_high_risk': False,
                'alerts': [],
                'return_rate_metrics': {}
            }
        
        try:
            # Retrieve return rate metrics based on configuration
            try:
                # First check the debtor account
                
                return_rate_metrics = self.db.get_ach_return_rate_metrics_actual(
                    transaction.get('debtor_account'),
                    transaction.get('creditor_account')
                )

                if not return_rate_metrics:
                    return {
                        'unauthorized_return_rate': 0.0,
                        'administrative_return_rate': 0.0,
                        'overall_return_rate': 0.0,
                        'total_transactions': 0,
                        'suspicious_transactions': 0
                    }


            except Exception as e:
                logger.error(f"Error retrieving ACH return rate metrics: {str(e)}")
                return_rate_metrics = {
                    'unauthorized_return_rate': 0.0,
                    'administrative_return_rate': 0.0,
                    'overall_return_rate': 0.0,
                    'return_code_counts': {}
                }
            
            # Define thresholds from configuration
            unauthorized_threshold = float(self.config_reader.get_property('ach.return_rate.unauthorized_threshold', '0.5'))
            administrative_threshold = float(self.config_reader.get_property('ach.return_rate.administrative_threshold', '3.0'))
            overall_threshold = float(self.config_reader.get_property('ach.return_rate.overall_threshold', '15.0'))
            
            # High-risk return codes (R05: Unauthorized, R07: Authorization Revoked, R10: Customer Advises Not Authorized)
            #Since we are checking for 3 codes, the threshold will always be 3, hence not in config.properties
            high_risk_code_threshold = int(self.config_reader.get_property('ach.return_rate.high_risk_code_threshold', '3'))
            
            # Prepare return rate metrics
            metrics = {
                'unauthorized_return_rate': return_rate_metrics.get('unauthorized_return_rate', 0),
                'administrative_return_rate': return_rate_metrics.get('administrative_return_rate', 0),
                'overall_return_rate': return_rate_metrics.get('overall_return_rate', 0),
                'return_code_counts': return_rate_metrics.get('return_code_counts', {})
            }
            
            # Initialize alerts list
            alerts = []
            is_high_risk = False
            
            # Check unauthorized return rate
            if metrics['unauthorized_return_rate'] > unauthorized_threshold:
                alerts.append({
                    'type': 'ACH_UNAUTHORIZED_RETURN_RATE_HIGH',
                    'severity': 'HIGH',
                    'details': f"Unauthorized return rate ({metrics['unauthorized_return_rate']}%) exceeds threshold of {unauthorized_threshold}%"
                })
                is_high_risk = True
            
            # Check administrative return rate
            if metrics['administrative_return_rate'] > administrative_threshold:
                alerts.append({
                    'type': 'ACH_ADMINISTRATIVE_RETURN_RATE_HIGH',
                    'severity': 'MEDIUM',
                    'details': f"Administrative return rate ({metrics['administrative_return_rate']}%) exceeds threshold of {administrative_threshold}%"
                })
                is_high_risk = True
            
            # Check overall return rate
            if metrics['overall_return_rate'] > overall_threshold:
                alerts.append({
                    'type': 'ACH_OVERALL_RETURN_RATE_HIGH',
                    'severity': 'CRITICAL',
                    'details': f"Overall return rate ({metrics['overall_return_rate']}%) exceeds threshold of {overall_threshold}%"
                })
                is_high_risk = True
            
            # Check specific high-risk return codes
            # R05-Unauthorized Debit
            # R07-Authorization Revoked by Customer
            # R10-Unauthorized Debit
            unauthorized_codes = ['R05', 'R07', 'R10']
            for code in unauthorized_codes:
                code_count = metrics['return_code_counts'].get(code, 0)
                if code_count >= high_risk_code_threshold:
                    code_description = self.ACH_RETURN_CODES.get(code, {}).get('text', f"Code {code}")
                    alerts.append({
                        'type': f'ACH_RETURN_CODE_{code}_HIGH',
                        'severity': 'HIGH',
                        'details': f"High frequency of {code_description} returns ({code_count}) exceeds threshold of {high_risk_code_threshold}"
                    })
                    is_high_risk = True
            
            # Check for account closed returns
            account_closed_codes = ['R02', 'R03']
            for code in account_closed_codes:
                code_count = metrics['return_code_counts'].get(code, 0)
                if code_count > 0:
                    code_description = self.ACH_RETURN_CODES.get(code, {}).get('text', f"Code {code}")
                    alerts.append({
                        'type': f'ACH_RETURN_CODE_{code}_DETECTED',
                        'severity': 'MEDIUM',
                        'details': f"Account issue detected: {code_description} ({code_count} occurrences)"
                    })
                    # Don't automatically mark as high risk, but flag for review

            # Check for fraud-specific return codes
            fraud_codes = ['R08', 'R11', 'R16', 'R29', 'R38', 'R39', 'R51']
            for code in fraud_codes:
                code_count = metrics['return_code_counts'].get(code, 0)
                if code_count > 0:
                    code_description = self.ACH_RETURN_CODES.get(code, {}).get('text', f"Code {code}")
                    alerts.append({
                        'type': f'ACH_FRAUD_INDICATOR_{code}',
                        'severity': 'HIGH',
                        'details': f"Potential fraud indicator: {code_description} ({code_count} occurrences)"
                    })
                    is_high_risk = True
            
            # Add return code information to metrics for reference
            significant_codes = []
            for code, count in metrics['return_code_counts'].items():
                if count > 0:
                    code_description = self.ACH_RETURN_CODES.get(code, {}).get('text', 'Unknown')
                    significant_codes.append({
                        'code': code, 
                        'description': code_description,
                        'count': count
                    })
            
            metrics['significant_return_codes'] = significant_codes
            
            return {
                'is_high_risk': is_high_risk,
                'alerts': alerts,
                'return_rate_metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"Error checking ACH return rates: {str(e)}", exc_info=True)
            return {
                'is_high_risk': False,
                'alerts': [],
                'return_rate_metrics': {}
            }
    
    def _process_compliance_checks(self, transaction: Dict, result: Dict) -> Dict:
        """Process all compliance checks with optimized list checking"""
        try:
            # Initialize default results structure
            compliance_results = {
                'whitelist': {
                    'is_whitelisted': False,
                    'matched_entry': None,
                    'match_type': None,
                    'risk_score': 0,
                    'risk_level': 'LOW',
                    'severity': 'LOW'
                },
                'blacklist': {
                    'is_blacklisted': False,
                    'matched_entry': None,
                    'match_type': None,
                    'risk_score': 0,
                    'risk_level': 'LOW',
                    'severity': 'LOW'
                },
                'watchlist': {
                    'is_watchlisted': False,
                    'matched_entry': None,
                    'match_type': None,
                    'risk_score': 0,
                    'risk_level': 'LOW',
                    'severity': 'LOW'
                },
                'ofac': {
                    'is_ofac_blocked': False,
                    'debtor_ofac_matches': [],
                    'creditor_ofac_matches': [],
                    'match_score': 0,
                    'severity': 'LOW'
                }
            }
            
            import time
            start_time = time.time()
            
            # OPTIMIZATION: Check all platform lists at once
            platform_list_start = time.time()
            platform_list_results = self._check_all_platform_lists(transaction)
            platform_list_time = time.time() - platform_list_start
            # self.logger.info(f"All platform lists check time: {platform_list_time:.4f}s")
            
            # Process whitelist results
            whitelist_check = platform_list_results['whitelist']
            if whitelist_check and whitelist_check.get('is_whitelisted'):
                # Handle whitelist match (early return case)
                compliance_results['whitelist'] = whitelist_check
                transaction['whitelist_check'] = whitelist_check
                
                # Get matched entry details for whitelist
                matched_entry = whitelist_check.get('matched_entry', {})
                match_type = whitelist_check.get('match_type', 'UNKNOWN')
                
                # Generate whitelist details
                whitelist_details = self._format_list_match_details(matched_entry)
                
                # Create a consolidated whitelist message
                whitelist_message = f"WHITELISTED: {match_type} match. " + \
                                (f"Details: {'; '.join(whitelist_details)}" if whitelist_details else "No additional details")
                
                # Update result with whitelist information
                result.update({
                    'final_risk_score': 0,
                    'whitelist_match': whitelist_message,
                    'bypass_reason': f"Whitelisted by {match_type} match",
                    'processing_status': 'BYPASSED'
                })

                # Insert processed message
                self.db.insert_processed_message(
                    transaction,
                    None,
                    result.get('final_risk_score', 0),
                    whitelist_message,
                    ''
                )
                
                # total_time = time.time() - start_time
                # self.logger.info(f"Total compliance checks time (early whitelist return): {total_time:.4f}s")
                return {'early_return': True, 'result': result}
            else:
                transaction['whitelist_check'] = whitelist_check or {}
            
            # Process blacklist results
            blacklist_check = platform_list_results['blacklist']
            if blacklist_check and blacklist_check.get('is_blacklisted'):
                compliance_results['blacklist'] = blacklist_check
                transaction['blacklist_check'] = blacklist_check
                
                # Update risk score
                if blacklist_check.get('risk_score'):
                    result['final_risk_score'] += float(blacklist_check.get('risk_score', 0))
                
                # Process blacklist alert for high risk entries
                if blacklist_check.get('risk_level') in ['CRITICAL', 'HIGH']:
                    self._process_blacklist_alert(blacklist_check, result)
            else:
                transaction['blacklist_check'] = blacklist_check or {}
            
            # Process watchlist results
            watchlist_check = platform_list_results['watchlist']
            if watchlist_check and watchlist_check.get('is_watchlisted'):
                compliance_results['watchlist'] = watchlist_check
                transaction['watchlist_check'] = watchlist_check
                
                # Update risk score
                if watchlist_check.get('risk_score'):
                    result['final_risk_score'] += float(watchlist_check.get('risk_score', 0))
                    
                # Process watchlist alert
                self._process_watchlist_alert(watchlist_check, result)
            else:
                transaction['watchlist_check'] = watchlist_check or {}
            
            # Process OFAC check
            # ofac_check_start = time.time()
            try:
                # Call the new generate_ofac_report method instead of check_transaction_names
                ofac_report = self.ofac_checker.generate_ofac_report(transaction)
                
                if ofac_report:
                    # Store both the raw results and formatted report
                    compliance_results['ofac'] = ofac_report['raw_results']
                    compliance_results['ofac_formatted_report'] = ofac_report['formatted_report']
                    
                    # Add both to the transaction object
                    transaction['ofac_check'] = ofac_report['raw_results']
                    transaction['ofac_report'] = ofac_report['formatted_report']
                    
                    # Update risk score based on OFAC match score from raw results
                    if ofac_report['raw_results'].get('match_score'):
                        result['final_risk_score'] += float(ofac_report['raw_results'].get('match_score', 0))
                    
                    # # Process OFAC alert if needed
                    # if ofac_report['raw_results'].get('is_ofac_blocked'):
                    #     self._process_ofac_alert(ofac_report['raw_results'], result)
                else:
                    transaction['ofac_check'] = compliance_results['ofac']
                    transaction['ofac_report'] = "OFAC check completed. No matches found."
            except Exception as e:
                self.logger.error(f"Error in OFAC check: {str(e)}")
                transaction['ofac_check'] = compliance_results['ofac']
                transaction['ofac_report'] = f"Error generating OFAC report: {str(e)}"
            # try:
            #     ofac_results = self.ofac_checker.check_transaction_names(transaction, result['final_risk_score'])
            #     if ofac_results:
            #         compliance_results['ofac'] = ofac_results
            #         transaction['ofac_check'] = ofac_results
                    
            #         # Update risk score based on OFAC
            #         if ofac_results.get('match_score'):
            #             result['final_risk_score'] += float(ofac_results.get('match_score', 0))
                        
            #         # # Process OFAC alert if needed
            #         # if ofac_results.get('is_ofac_blocked'):
            #         #     self._process_ofac_alert(ofac_results, result)
            #     else:
            #         transaction['ofac_check'] = compliance_results['ofac']
            # except Exception as e:
            #     self.logger.error(f"Error in OFAC check: {str(e)}")
            #     transaction['ofac_check'] = compliance_results['ofac']

            # Calculate Compliance Risk
            try:
                compliance_risk, compliance_alerts = self.calculate_compliance_alert_risk(
                    ofac_report['raw_results'] if ofac_report else {},
                    watchlist_check,
                    blacklist_check
                )

                if compliance_alerts:
                    # result['is_suspicious'] = True
                    result['final_risk_score'] += compliance_risk
                    result['risk_factors']['compliance_alerts'].extend(compliance_alerts)
                    
                    # Add alert reasons to main alerts list
                    for alert in compliance_alerts:
                        if isinstance(alert, dict):
                            result['alerts'].append(alert)
                        elif alert:
                            result['alerts'].append(str(alert))

                result['compliance_score'] = result['final_risk_score']

                # Add compliance information to risk factors
                result['risk_factors'].update({
                    'ofac_matches': bool(ofac_report['raw_results'].get('is_ofac_blocked') if ofac_report else False),
                    'watchlist_matches': bool(watchlist_check.get('is_watchlisted')),
                    'blacklist_matches': bool(blacklist_check.get('is_blacklisted')),
                    'compliance_risk_score': compliance_risk
                })

                # Process compliance alerts
                try:
                    severity=""
                    compliance_alert_text = ""
                    consolidated_alerts = []
                    # Get the formatted OFAC report if available
                    ofac_formatted_report = transaction.get('ofac_report', '') 

                    if 'risk_factors' in result and 'compliance_alerts' in result['risk_factors'] and result['risk_factors']['compliance_alerts']:
                        
                        # Extract compliance alert messages
                        compliance_alerts = []
                        for alert in result['risk_factors']['compliance_alerts']:
                            if isinstance(alert, dict):
                                severity = alert.get('severity')
                                alert_text = alert.get('consolidated_message') or alert.get('reason') or alert.get('details', '')
                                if alert_text:
                                    compliance_alerts.append(alert_text)
                            elif alert:
                                severity = alert.get('severity')
                                compliance_alerts.append(str(alert))
                        
                        # Join alert messages
                        alert_messages = " | ".join(filter(None, compliance_alerts))

                        # Use the formatted OFAC report as the main alert text if available
                        if ofac_formatted_report and 'ofac_matches' in result['risk_factors'] and result['risk_factors']['ofac_matches']:
                            compliance_alert_text = ofac_formatted_report
                        else:
                            compliance_alert_text = alert_messages

                        if result['compliance_score'] >= float(self.config_reader.get_property('risk.compliance.threshold', '60')):

                            # Insert compliance-specific fraud alert - using the transaction from the loop
                            alert_id = self.db.insert_fraud_alert(
                                transaction,
                                result.get('compliance_score',0) ,
                                compliance_alert_text,
                                0,  # entity_id
                                "Compliance Alert",  # Specific fraud type for compliance alerts
                                "Screening",
                                severity 
                                # "HIGH"  # Compliance alerts are typically high severity
                            )

                            # Use the formatted OFAC report as the alert text for the processed message
                            detailed_alert_text = ofac_formatted_report if ofac_formatted_report else alert_messages

                            # Insert processed message - using the transaction from the loop
                            self.db.insert_processed_message(
                                transaction,
                                alert_id,
                                result.get('compliance_score',0) ,
                                detailed_alert_text,
                                ''
                            )

                except Exception as e:
                    self.logger.error(f"Error in compliance alert database operations : {str(e)}")
                
                return result

            except Exception as e:
                    self.logger.error(f"Error in calculate_compliance_alert_risk : {str(e)}")
                
        except Exception as e:
            self.logger.error(f"Error in compliance checks: {str(e)}")
            # Ensure transaction has at least empty results
            transaction['whitelist_check'] = transaction.get('whitelist_check', {})
            transaction['blacklist_check'] = transaction.get('blacklist_check', {})
            transaction['watchlist_check'] = transaction.get('watchlist_check', {})
            transaction['ofac_check'] = transaction.get('ofac_check', {})
            return result


    def _fetch_entity_data_batch(self, transaction: Dict) -> Dict:
        """
        Batch fetch all entity data needed for compliance checks - Simplified Version
        
        Returns:
            Dict with all entity data needed for checks
        """
        try:
            result = {
                'debtor': {
                    'industry_code': None,
                    'phones': [],
                    'emails': [],
                    'ssns': [],
                    'taxids': []
                },
                'creditor': {
                    'industry_code': None,
                    'phones': [],
                    'emails': [],
                    'ssns': [],
                    'taxids': []
                }
            }
            
            # Make a single DB connection
            self.db.reconnect()
            
            # Just get industry codes for now to test connection
            debtor_account = transaction.get('debtor_account')
            if debtor_account:
                try:
                    result['debtor']['industry_code'] = self.db.get_industry_code(debtor_account)
                except Exception as e:
                    self.logger.error(f"Error fetching debtor industry code: {str(e)}")
            
            creditor_account = transaction.get('creditor_account_number')
            if creditor_account:
                try:
                    result['creditor']['industry_code'] = self.db.get_industry_code(creditor_account)
                except Exception as e:
                    self.logger.error(f"Error fetching creditor industry code: {str(e)}")
                    
            return result
            
        except Exception as e:
            self.logger.error(f"Error in batch fetching entity data: {str(e)}")
            return {
                'debtor': {'industry_code': None, 'phones': [], 'emails': [], 'ssns': [], 'taxids': []},
                'creditor': {'industry_code': None, 'phones': [], 'emails': [], 'ssns': [], 'taxids': []}
            }
    
    

    def _check_platform_list(self, transaction: Dict, list_type: ListType, entities: List[Dict], current_time) -> Dict:
        """
        Optimized database check for platform lists with robust error handling
        """
        try:
            # Initialize default result based on list type
            if list_type == ListType.WHITELIST:
                result = {
                    'is_whitelisted': False,
                    'matched_entry': None,
                    'match_type': None,
                    'reason': None
                }
                key_prefix = 'is_whitelisted'
            elif list_type == ListType.BLACKLIST:
                result = {
                    'is_blacklisted': False,
                    'matched_entry': None,
                    'match_type': None,
                    'risk_score': 0,
                    'risk_level': 'LOW',
                    'severity': 'LOW'
                }
                key_prefix = 'is_blacklisted'
            elif list_type == ListType.WATCHLIST:
                result = {
                    'is_watchlisted': False,
                    'matched_entry': None,
                    'match_type': None,
                    'risk_score': 0,
                    'risk_level': 'LOW'
                }
                key_prefix = 'is_watchlisted'
            else:
                return {}
            
            # Log what we're checking
            self.logger.debug(f"Checking {len(entities)} entities against {list_type.value}")
            
            # Ensure entities is not empty
            if not entities:
                return result
                
            # Print the first entity for debugging
            self.logger.debug(f"First entity type: {type(entities[0])}")
            if isinstance(entities[0], dict):
                self.logger.debug(f"First entity keys: {entities[0].keys()}")
            else:
                self.logger.debug(f"First entity value: {entities[0]}")
                    
            # Ultimate fallback - use the original method directly on the full transaction
            # This bypasses any entity parsing issues
            return self._check_platform_list(transaction, list_type)
                
        except Exception as e:
            self.logger.error(f"Error in optimized database check for platform list ({list_type.value}): {str(e)}")
            # Ultimate fallback - use the original method
        
    def _get_value_from_record(self, record, key, default=None):
        """
        Safely extract a value from a database record, handling different record types
        """
        try:
            if isinstance(record, dict):
                return record.get(key, default)
            elif hasattr(record, key):  # For namedtuple-like objects
                return getattr(record, key, default)
            elif hasattr(record, '_fields'):  # For namedtuple with explicitly defined fields
                try:
                    idx = record._fields.index(key)
                    return record[idx]
                except (ValueError, IndexError):
                    return default
            else:
                # For list/tuple type records, try to find key in the column names
                if hasattr(self.db, 'get_column_names'):
                    columns = self.db.get_column_names()
                    if key in columns:
                        idx = columns.index(key)
                        return record[idx]
                return default
        except Exception as e:
            self.logger.debug(f"Error extracting {key} from record: {str(e)}")
            return default

    def _is_entry_expired(self, expiry_date, current_time):
        """Check if an entry is expired based on its expiry date"""
        if not expiry_date:
            return False
            
        try:
            # Handle different date formats
            if isinstance(expiry_date, datetime.datetime):
                return expiry_date < current_time
            elif isinstance(expiry_date, datetime.date):
                # Convert date to datetime for comparison
                expiry_datetime = datetime.datetime.combine(expiry_date, datetime.time.min)
                if expiry_datetime.tzinfo is None:
                    expiry_datetime = expiry_datetime.replace(tzinfo=datetime.timezone.utc)
                return expiry_datetime < current_time
            elif isinstance(expiry_date, str):
                try:
                    # Try datetime format first
                    expiry = datetime.datetime.strptime(expiry_date, "%Y-%m-%d %H:%M:%S")
                    if expiry.tzinfo is None:
                        expiry = expiry.replace(tzinfo=datetime.timezone.utc)
                    return expiry < current_time
                except (ValueError, TypeError):
                    # Fallback to date format
                    try:
                        expiry = datetime.datetime.strptime(expiry_date, "%Y-%m-%d")
                        if expiry.tzinfo is None:
                            expiry = expiry.replace(tzinfo=datetime.timezone.utc)
                        return expiry < current_time
                    except (ValueError, TypeError):
                        # If all parsing fails, assume not expired
                        return False
        except Exception:
            # If any error occurs during comparison, assume not expired
            return False
            
        return False

    def _is_within_time_window(self, entry, current_time):
        """Check if current time is within the entry's time window"""
        start_time = entry.get('start_time')
        end_time = entry.get('end_time')
        
        if not start_time and not end_time:
            return True  # No time restrictions
            
        try:
            # Handle start_time
            if start_time:
                if isinstance(start_time, str):
                    try:
                        start_time = datetime.datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                    except (ValueError, AttributeError):
                        try:
                            start_time = datetime.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
                        except (ValueError, TypeError):
                            start_time = None
                            
                if isinstance(start_time, datetime.datetime) and start_time.tzinfo is None:
                    start_time = start_time.replace(tzinfo=datetime.timezone.utc)
                    
                if start_time and current_time < start_time:
                    return False
                    
            # Handle end_time
            if end_time:
                if isinstance(end_time, str):
                    try:
                        end_time = datetime.datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                    except (ValueError, AttributeError):
                        try:
                            end_time = datetime.datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
                        except (ValueError, TypeError):
                            end_time = None
                            
                if isinstance(end_time, datetime.datetime) and end_time.tzinfo is None:
                    end_time = end_time.replace(tzinfo=datetime.timezone.utc)
                    
                if end_time and current_time > end_time:
                    return False
        except Exception:
            # If any error occurs during comparison, assume within time window
            return True
            
        return True

    def _format_list_match_details(self, matched_entry):
        """Format match details consistently"""
        details = []
        if matched_entry:
            if matched_entry.get('entity_name'):
                details.append(f"Name: {matched_entry.get('entity_name')}")
            if matched_entry.get('identifier'):
                details.append(f"ID: {matched_entry.get('identifier')}")
            if matched_entry.get('category'):
                details.append(f"Category: {matched_entry.get('category')}")
            if matched_entry.get('risk_level'):
                details.append(f"Risk Level: {matched_entry.get('risk_level').upper()}")
            if matched_entry.get('notes'):
                details.append(f"Notes: {matched_entry.get('notes')}")
        return details

    def _process_blacklist_alert(self, blacklist_check, result):
        """Process blacklist alert consistently"""
        matched_entry = blacklist_check.get('matched_entry', {})
        match_type = blacklist_check.get('match_type', 'UNKNOWN')
        
        # Create detailed information
        blacklist_details = self._format_list_match_details(matched_entry)
        
        # Create a consolidated message
        blacklist_message = f"BLACKLISTED: {match_type} match with {blacklist_check.get('risk_level')} risk. " + \
                            (f"Details: {'; '.join(blacklist_details)}" if blacklist_details else "No additional details")
        
        # Update result with information
        result.update({
            'blacklist_match': blacklist_message,
            'processing_status': 'REJECTED',
            'rejection_reason': f"Entity is blacklisted with {blacklist_check.get('risk_level')} risk level"
        })
        
        # Add alert to results
        blacklist_alert = {
            'type': 'BLACKLIST_MATCH',
            'severity': blacklist_check.get('severity', 'HIGH'),
            'reason': f"Transaction involves blacklisted {match_type}",
            'details': {
                'entity': matched_entry.get('entity_name', ''),
                'identifier': matched_entry.get('identifier', ''),
                'risk_score': blacklist_check.get('risk_score', 0),
                'risk_level': blacklist_check.get('risk_level', 'HIGH')
            },
            'consolidated_message': blacklist_message
        }
        
        if 'risk_factors' not in result:
            result['risk_factors'] = {}
        if 'compliance_alerts' not in result['risk_factors']:
            result['risk_factors']['compliance_alerts'] = []
        
        result['risk_factors']['compliance_alerts'].append(blacklist_alert)
        
        if 'alerts' not in result:
            result['alerts'] = []
        result['alerts'].append(blacklist_message)

    def _process_watchlist_alert(self, watchlist_check, result):
        """Process watchlist alert consistently"""
        matched_entry = watchlist_check.get('matched_entry', {})
        match_type = watchlist_check.get('match_type', 'UNKNOWN')
        
        # Create detailed information
        watchlist_details = self._format_list_match_details(matched_entry)
        
        # Create a consolidated message
        watchlist_message = f"WATCHLISTED: {match_type} match with {watchlist_check.get('risk_level')} risk. " + \
                            (f"Details: {'; '.join(watchlist_details)}" if watchlist_details else "No additional details")
        
        # Add alert to results
        watchlist_alert = {
            'type': 'WATCHLIST_MATCH',
            'severity': watchlist_check.get('severity', 'MEDIUM'),
            'reason': f"Transaction involves watchlisted {match_type}",
            'details': {
                'entity': matched_entry.get('entity_name', ''),
                'identifier': matched_entry.get('identifier', ''),
                'risk_score': watchlist_check.get('risk_score', 0),
                'risk_level': watchlist_check.get('risk_level', 'MEDIUM')
            },
            'consolidated_message': watchlist_message
        }
        
        if 'risk_factors' not in result:
            result['risk_factors'] = {}
        if 'compliance_alerts' not in result['risk_factors']:
            result['risk_factors']['compliance_alerts'] = []
        
        result['risk_factors']['compliance_alerts'].append(watchlist_alert)
        
        if 'alerts' not in result:
            result['alerts'] = []
        result['alerts'].append(watchlist_message)

    def _process_ofac_alert(self, ofac_results, result):
        """Process OFAC alert consistently"""
        # Create detailed information for OFAC
        ofac_details = []
        if ofac_results.get('debtor_ofac_matches'):
            ofac_details.append(f"Debtor matches: {len(ofac_results.get('debtor_ofac_matches'))}")
        if ofac_results.get('creditor_ofac_matches'):
            ofac_details.append(f"Creditor matches: {len(ofac_results.get('creditor_ofac_matches'))}")
        if ofac_results.get('match_score'):
            ofac_details.append(f"Match score: {ofac_results.get('match_score')}")
        
        # Create a consolidated message
        ofac_message = f"OFAC BLOCKED: Potential OFAC matches found. " + \
                    (f"Details: {'; '.join(ofac_details)}" if ofac_details else "No additional details")
        
        # Update result with information
        result.update({
            'ofac_match': ofac_message,
            'processing_status': 'REJECTED',
            'rejection_reason': "Potential OFAC match detected"
        })
        
        # Add alert to results
        ofac_alert = {
            'type': 'OFAC_MATCH',
            'severity': ofac_results.get('severity', 'CRITICAL'),
            'reason': "Transaction involves potential OFAC match",
            'details': {
                'debtor_matches': ofac_results.get('debtor_ofac_matches', []),
                'creditor_matches': ofac_results.get('creditor_ofac_matches', []),
                'match_score': ofac_results.get('match_score', 0)
            },
            'consolidated_message': ofac_message
        }
        
        if 'risk_factors' not in result:
            result['risk_factors'] = {}
        if 'compliance_alerts' not in result['risk_factors']:
            result['risk_factors']['compliance_alerts'] = []
        
        result['risk_factors']['compliance_alerts'].append(ofac_alert)
        
        if 'alerts' not in result:
            result['alerts'] = []
        result['alerts'].append(ofac_message)

    def _check_platform_list(self, transaction: Dict, list_type: ListType) -> Dict:
        """
        Check if transaction parties are in the specified platform list
        
        Parameters:
        - transaction: Transaction data
        - list_type: Type of list to check (whitelist, blacklist, watchlist)
        
        Returns:
        - Dictionary with match results
        """
        try:
            # Initialize default result based on list type
            if list_type == ListType.WHITELIST:
                result = {
                    'is_whitelisted': False,
                    'matched_entry': None,
                    'match_type': None,
                    'reason': None
                }
                key_prefix = 'is_whitelisted'
            elif list_type == ListType.BLACKLIST:
                result = {
                    'is_blacklisted': False,
                    'matched_entry': None,
                    'match_type': None,
                    'risk_score': 0,
                    'risk_level': 'LOW',
                    'severity': 'LOW'
                }
                key_prefix = 'is_blacklisted'
            elif list_type == ListType.WATCHLIST:
                result = {
                    'is_watchlisted': False,
                    'matched_entry': None,
                    'match_type': None,
                    'risk_score': 0,
                    'risk_level': 'LOW'
                }
                key_prefix = 'is_watchlisted'
            else:
                return {}
            
            # Check if we have PlatformListManager initialized
            if not hasattr(self, 'platform_list_manager') or self.platform_list_manager is None:
                # This service might be in a transition phase where PlatformListManager is not yet set up
                self.logger.warning(f"PlatformListManager not initialized, skipping {list_type.value} check")
                return result
            
            # Extract entities to check from transaction
            entities = []

            debtor_industry_code = self.db.get_industry_code(transaction.get('debtor_account'))

            # If this returns a tuple with one element, extract the value:
            if isinstance(debtor_industry_code, tuple) and len(debtor_industry_code) > 0:
                debtor_industry_code = debtor_industry_code[0]

            creditor_industry_code = self.db.get_industry_code(transaction.get('debtor_account'))

            # If this returns a tuple with one element, extract the value:
            if isinstance(creditor_industry_code, tuple) and len(creditor_industry_code) > 0:
                creditor_industry_code = creditor_industry_code[0]


            # Get current time in UTC for time-based filtering
            from datetime import datetime, timezone
            current_time = datetime.now(timezone.utc)
            
            # Add debtor/source account details if available
            if transaction.get('debtor_name'):
                entities.append({
                    'entity_name': transaction.get('debtor_name', ''),
                    'identifier': transaction.get('debtor_name', ''),
                    'entity_type': 'individual',
                    'role': 'debtor',
                    'country': transaction.get('debtor_country'),
                    'industry_code': debtor_industry_code,
                    'payment_channel':transaction.get('channel')
                })
                
                # Also add debtor's bank if available
                if transaction.get('debtor_account'):
                    entities.append({
                        'entity_name': transaction.get('debtor_account', ''),
                        'identifier':  transaction.get('debtor_account',''),
                        'entity_type': 'financial_institution',
                        'role': 'debtor_bank',
                        'country': transaction.get('debtor_country'),
                        'industry_code': debtor_industry_code,
                        'payment_channel':transaction.get('channel')
                    })

                if transaction.get('debtor_account_routing_number'):
                    entities.append({
                        'entity_name': f"Routing: {transaction.get('debtor_account_routing_number')}",
                        'identifier': transaction.get('debtor_account_routing_number', ''),
                        'information_type': 'bank_routing_number',  # Specify this is a routing number
                        'entity_type': 'financial_institution',
                        'role': 'debtor_bank',
                        'country': transaction.get('debtor_country'),
                        'industry_code': debtor_industry_code,
                        'payment_channel':transaction.get('channel')
                    })

                if transaction.get('debtor_country'):
                    entities.append({
                        'entity_name': f"Debtor Country: {transaction.get('debtor_country')}",
                        'identifier': transaction.get('debtor_country', ''),
                        'information_type': 'debtor_country',  # Specify this is a debtor country
                        'entity_type': 'country',
                        'role': 'debtor_bank',
                        'country': transaction.get('debtor_country'),
                        'industry_code': debtor_industry_code,
                        'payment_channel':transaction.get('channel')
                    })


            #Try to get other identifiers
        
            phone_numbers = self.db.get_entity_phone_numbers(transaction.get('debtor_account'))  or []

            for phone_info in phone_numbers:
                
                if phone_info:
                    entities.append({
                        'entity_name': phone_info,
                        'identifier': phone_info,
                        'information_type': 'phone',  
                        'entity_type': 'individual',
                        'role': 'debtor_bank',
                        'country': transaction.get('debtor_country'),
                        'industry_code': debtor_industry_code,
                        'payment_channel':transaction.get('channel')
                    })
            
            email_addresses = self.db.get_entity_email(transaction.get('debtor_account'))  or []

            for email in email_addresses:
                
                if email:
                    entities.append({
                        'entity_name': email,
                        'identifier': email,
                        'information_type': 'email',  
                        'entity_type': 'individual',
                        'role': 'debtor_bank',
                        'country': transaction.get('debtor_country'),
                        'industry_code': debtor_industry_code,
                        'payment_channel':transaction.get('channel')
                    })

            ssns = self.db.get_entity_ssn(transaction.get('debtor_account'))  or []

            for ssn_val in ssns:
                
                if ssn_val:
                    entities.append({
                        'entity_name': ssn_val,
                        'identifier': ssn_val,
                        'information_type': 'SSN',  
                        'entity_type': 'individual',
                        'role': 'debtor_bank',
                        'country': transaction.get('debtor_country'),
                        'industry_code': debtor_industry_code,
                        'payment_channel':transaction.get('channel')
                    })

            taxids = self.db.get_entity_taxid(transaction.get('debtor_account'))  or []

            for taxid_val in taxids:
                
                if taxid_val:
                    entities.append({
                        'entity_name': taxid_val,
                        'identifier': taxid_val,
                        'information_type': 'TaxId',  
                        'entity_type': 'organization',
                        'role': 'debtor_bank',
                        'country': transaction.get('debtor_country'),
                        'industry_code': debtor_industry_code,
                        'payment_channel':transaction.get('channel')
                    })
                        
            # Add creditor/destination account details if available
            if transaction.get('creditor_name'):
                entities.append({
                    'entity_name': transaction.get('creditor_name', ''),
                    'identifier': transaction.get('creditor_account', ''),
                    'entity_type': 'individual',
                    'role': 'creditor',
                    'country': transaction.get('creditor_country'),
                    'industry_code': creditor_industry_code,
                    'payment_channel':transaction.get('channel')
                })
                
                # Also add creditor's bank if available
                if transaction.get('creditor_account_number'):
                    entities.append({
                        'entity_name': transaction.get('creditor_account_number', ''),
                        'identifier': transaction.get('creditor_account_number',''),
                        'entity_type': 'financial_institution',
                        'role': 'creditor_bank',
                        'country': transaction.get('creditor_country'),
                        'industry_code': creditor_industry_code,
                    'payment_channel':transaction.get('channel')
                    })

                # If routing number exists, add it as a separate entity
                if transaction.get('creditor_account_routing_number'):
                    entities.append({
                        'entity_name': f"Routing: {transaction.get('creditor_account_routing_number')}",
                        'identifier': transaction.get('creditor_account_routing_number', ''),
                        'information_type': 'bank_routing_number',
                        'entity_type': 'financial_institution',
                        'role': 'creditor_bank',
                        'country': transaction.get('creditor_country'),
                        'industry_code': creditor_industry_code,
                    'payment_channel':transaction.get('channel')
                    })


                if transaction.get('creditor_country'):
                    entities.append({
                        'entity_name': f"Creditor Country: {transaction.get('creditor_country')}",
                        'identifier': transaction.get('creditor_country', ''),
                        'information_type': 'creditor_country',  
                        'entity_type': 'country',
                        'role': 'creditor_bank',
                        'country': transaction.get('creditor_country'),
                        'industry_code': creditor_industry_code,
                    'payment_channel':transaction.get('channel')
                    })

            phone_numbers = self.db.get_entity_phone_numbers(transaction.get('creditor_account_number')) or []

            for phone_info in phone_numbers:
                
                if phone_info:
                    entities.append({
                        'entity_name': phone_info,
                        'identifier': phone_info,
                        'information_type': 'phone',  
                        'entity_type': 'individual',
                        'role': 'debtor_bank',
                        'country': transaction.get('creditor_country'),
                        'industry_code': creditor_industry_code,
                'payment_channel':transaction.get('channel')
                    })          
            
            email_addresses = self.db.get_entity_email(transaction.get('creditor_account_number'))  or []

            for email in email_addresses:
                
                if email:
                    entities.append({
                        'entity_name': email,
                        'identifier': email,
                        'information_type': 'email',  
                        'entity_type': 'individual',
                        'role': 'debtor_bank',
                        'country': transaction.get('creditor_country'),
                        'industry_code': creditor_industry_code,
                        'payment_channel':transaction.get('channel')
                    })

            ssns = self.db.get_entity_ssn(transaction.get('creditor_account_number')) or []

            for ssn_val in ssns:
                
                if ssn_val:
                    entities.append({
                        'entity_name': ssn_val,
                        'identifier': ssn_val,
                        'information_type': 'SSN',  
                        'entity_type': 'individual',
                        'role': 'debtor_bank',
                        'country': transaction.get('creditor_country'),
                        'industry_code': creditor_industry_code,
                        'payment_channel':transaction.get('channel')
                    })

            taxids = self.db.get_entity_taxid(transaction.get('creditor_account_number'))  or []

            for taxid_val in taxids:
                
                if taxid_val:
                    entities.append({
                        'entity_name': taxid_val,
                        'identifier': taxid_val,
                        'information_type': 'TaxId',  
                        'entity_type': 'organization',
                        'role': 'debtor_bank',
                        'country': transaction.get('creditor_country'),
                        'industry_code': creditor_industry_code,
                        'payment_channel':transaction.get('channel')
                    })
                    
            # For each entity, check against platform list
            for entity in entities:
                if not entity['entity_name'] or not entity['identifier']:
                    continue
                    
                # First filter entities by metadata (country, industry_code, entity_type)
                # This should return potential entity IDs that match these metadata criteria
                filtered_entities = self.filter_entities(
                    list_type=list_type,
                    country=entity.get('country'),
                    debtor_industry_code=debtor_industry_code,
                    creditor_industry_code=creditor_industry_code,
                    entity_type=entity.get('entity_type'),
                    payment_channel=entity.get('payment_channel'),
                    current_time=current_time
                ) or []

                if not filtered_entities:
                    continue

                # For each potential entity match, check if any identifiers match
                for entity_id in filtered_entities:
                    # Check if the identifier matches any information values for this entity
                    match = self.match_identifier(
                        entity_id=entity_id,
                        identifier=entity['identifier'],
                        information_type=entity.get('information_type')
                    )
                    
                    if match:
                        # Get full entity details for the matched entity
                        entity_details = self.get_entity_details(entity_id)
                        
                        if entity_details:
                            # Check time constraints using UTC time comparison
                            start_time = entity_details.get('start_time')
                            end_time = entity_details.get('end_time')
                            
                            # Ensure times are in UTC for comparison
                            if start_time and not start_time.tzinfo:
                                start_time = start_time.replace(tzinfo=datetime.timezone.utc)
                            if end_time and not end_time.tzinfo:
                                end_time = end_time.replace(tzinfo=datetime.timezone.utc)
                            
                            # Skip if current UTC time is outside the defined time window
                            if start_time and end_time:
                                if not (start_time <= current_time <= end_time):
                                    continue
                            elif start_time and not end_time:
                                if current_time < start_time:
                                    continue
                            elif end_time and not start_time:
                                if current_time > end_time:
                                    continue

                            result[key_prefix] = True
                            result['matched_entry'] = {
                                'entity_id': entity_id,
                                'entity_name': entity['entity_name'],
                                'identifier': entity['identifier'],
                                'entity_type': entity_details.get('entity_type')
                            }
                            result['match_type'] = f"{entity['role'].upper()}_MATCH"
                            
                            # Add additional details based on list type
                            if list_type in [ListType.BLACKLIST, ListType.WATCHLIST]:
                                result['risk_score'] = float(entity_details.get('risk_score', 0))
                                result['risk_level'] = entity_details.get('risk_level', 'LOW').upper()
                                
                                # For blacklist, set severity based on risk level
                                if list_type == ListType.BLACKLIST:
                                    if entity_details.get('risk_level') == 'critical':
                                        result['severity'] = 'CRITICAL'
                                    elif entity_details.get('risk_level') == 'high':
                                        result['severity'] = 'HIGH'
                                    elif entity_details.get('risk_level') == 'medium':
                                        result['severity'] = 'MEDIUM'
                            
                            # Exit early on first match
                            return result
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in platform list check ({list_type.value}): {str(e)}")
            return result
        
    def filter_entities(self, list_type, country=None, debtor_industry_code=None, creditor_industry_code=None, entity_type=None,  payment_channel=None, current_time=None):
        """
        Filter platform list entities by metadata before checking identifiers
        
        Returns a list of entity IDs that match the filter criteria
        """
        self.db.reconnect()
        query = "SELECT DISTINCT e.list_entity_id FROM platform_list_entities e JOIN platform_list_items i ON e.list_entity_id = i.list_entity_id WHERE i.list_type = %s"
        params = [list_type.value]
        
        if country:
            query += " AND (e.country = %s OR e.country IS NULL)"
            params.append(country)
        
        # Handle industry codes - match if entity's industry code matches either debtor or creditor industry code
        industry_conditions = []
        if debtor_industry_code:
            industry_conditions.append("e.industry_code = %s")
            params.append(debtor_industry_code)
        
        if creditor_industry_code and creditor_industry_code != debtor_industry_code:
            industry_conditions.append("e.industry_code = %s")
            params.append(creditor_industry_code)
        
        if industry_conditions:
            industry_conditions.append("e.industry_code IS NULL")
            query += " AND (" + " OR ".join(industry_conditions) + ")"
            
        if entity_type:
            query += " AND e.entity_type = %s"
            params.append(entity_type)

        # Add time window filtering if current_time is provided
        if current_time:
            time_conditions = [
                "(e.start_time IS NULL AND e.end_time IS NULL)",  # No time restrictions
                "(e.start_time IS NOT NULL AND e.end_time IS NOT NULL AND %s BETWEEN e.start_time AND e.end_time)",  # Within specified window
                "(e.start_time IS NOT NULL AND e.end_time IS NULL AND e.start_time <= %s)",  # After start time, no end
                "(e.start_time IS NULL AND e.end_time IS NOT NULL AND %s <= e.end_time)"  # Before end time, no start
            ]
            query += " AND (" + " OR ".join(time_conditions) + ")"
            params.extend([current_time, current_time, current_time])  # Add current_time 3 times for the conditions
        
        
        # Execute query
        results = self.db.execute_query(query, params)

        # Extract entity IDs
        return [row['list_entity_id'] for row in results]
    
    def match_identifier(self, entity_id, identifier, information_type=None):
        """
        Check if a specific identifier matches items for a given entity
        
        Returns True if a match is found, False otherwise
        """
        query = "SELECT * FROM platform_list_items WHERE list_entity_id = %s AND information_value = %s"
        params = [entity_id, identifier]
        
        if information_type:
            query += " AND information_type = %s"
            params.append(information_type)
        
        # Execute query
        with self.db.cursor() as cursor:
            cursor.execute(query, params)
            result = cursor.fetchone()
        
        return result is not None

    def get_entity_details(self, entity_id):
        """
        Get full details for an entity by ID
        
        Returns entity details as a dictionary
        """
        query = "SELECT * FROM platform_list_entities WHERE list_entity_id = %s"
        
        # Execute query
        with self.db.cursor() as cursor:
            cursor.execute(query, [entity_id])
            result = cursor.fetchone()
        
        return result

    def calculate_compliance_alert_risk(self, ofac_results: Dict, watchlist_results: Dict, blacklist_results: Dict = None) -> Tuple[float, List]:
        """
        Calculate compliance risk and generate alerts based on compliance check results.
        
        Parameters:
        - ofac_results: Results from OFAC check
        - watchlist_results: Results from watchlist check
        - blacklist_results: Results from blacklist check 
        
        Returns:
        - Tuple of (risk_score, alerts)
        """
        risk_score = 0
        alerts = []
        
        # Process OFAC results
        if ofac_results and ofac_results.get('is_ofac_blocked'):
            # OFAC hit is highest risk - set a high score
            ofac_score = float(self.config_reader.get_property('risk.ofac.match.score', '90'))
            risk_score += ofac_score
            
            # Create OFAC alert
            ofac_alert = {
                'type': 'OFAC_MATCH',
                'severity': 'CRITICAL',
                'reason': "Transaction involves possible sanctioned individual or entity",
                'details': {
                    'debtor_matches': ofac_results.get('debtor_ofac_matches', []),
                    'creditor_matches': ofac_results.get('creditor_ofac_matches', []),
                    'match_score': ofac_results.get('match_score', 0)
                },
                'consolidated_message': f"OFAC MATCH: Transaction involves sanctioned individual or entity. Match score: {ofac_results.get('match_score', 0)}"
            }
            alerts.append(ofac_alert)
        
        # Process blacklist results (new logic)
        if blacklist_results and blacklist_results.get('is_blacklisted'):
            # Blacklist hit is high risk
            blacklist_score = float(self.config_reader.get_property('risk.blacklist.match.score', '85'))
            risk_score += blacklist_score
            
            matched_entry = blacklist_results.get('matched_entry', {})
            match_type = blacklist_results.get('match_type', 'ENTITY')
            
            # Create detailed information
            match_details = []
            if matched_entry:
                if matched_entry.get('entity_name'):
                    match_details.append(f"Name: {matched_entry.get('entity_name')}")
                if matched_entry.get('identifier'):
                    match_details.append(f"ID: {matched_entry.get('identifier')}")
                if matched_entry.get('category'):
                    match_details.append(f"Category: {matched_entry.get('category')}")
                if matched_entry.get('risk_level'):
                    match_details.append(f"Risk Level: {matched_entry.get('risk_level').upper()}")
                if matched_entry.get('notes'):
                    match_details.append(f"Notes: {matched_entry.get('notes')}")
            
            # Create blacklist alert
            blacklist_alert = {
                'type': 'BLACKLIST_MATCH',
                'severity': blacklist_results.get('severity', 'HIGH'),
                'reason': f"Transaction involves blacklisted {match_type}",
                'details': {
                    'entity': matched_entry.get('entity_name', ''),
                    'identifier': matched_entry.get('identifier', ''),
                    'risk_score': blacklist_results.get('risk_score', 0),
                    'risk_level': blacklist_results.get('risk_level', 'HIGH')
                },
                'consolidated_message': f"BLACKLISTED: {match_type} match. " + 
                                        (f"Details: {'; '.join(match_details)}" if match_details else "No additional details")
            }
            alerts.append(blacklist_alert)
        
        # Process watchlist results
        if watchlist_results and watchlist_results.get('is_watchlisted'):
            # Watchlist hit - set a medium-high score
            watchlist_score = float(self.config_reader.get_property('risk.watchlist.match.score', '70'))
            
            # If risk score is available from platform list, use the higher one
            if watchlist_results.get('risk_score'):
                watchlist_score = max(watchlist_score, float(watchlist_results.get('risk_score', 0)))
                
            risk_score += watchlist_score
            
            matched_entry = watchlist_results.get('matched_entry', {})
            match_type = watchlist_results.get('match_type', 'ENTITY')
            
            # Create watchlist alert
            watchlist_alert = {
                'type': 'WATCHLIST_MATCH',
                'severity': 'HIGH',
                'reason': f"Transaction involves watchlisted {match_type}",
                'details': {
                    'entity': matched_entry.get('entity_name', ''),
                    'identifier': matched_entry.get('identifier', ''),
                    'risk_level': watchlist_results.get('risk_level', 'MEDIUM'),
                    'category': matched_entry.get('category', '')
                },
                'consolidated_message': f"WATCHLIST: {match_type} match. Entity: {matched_entry.get('entity_name', 'Unknown')}"
            }
            alerts.append(watchlist_alert)
        
        return risk_score, alerts



    def _update_transaction_pain001(self, transaction: Dict, pmt_info: Dict, tx: Dict) -> None:
        """Update transaction with PAIN.001 specific information"""
        try:
            # Update amount and currency
            if 'amount' in tx:
                if isinstance(tx['amount'], dict):
                    transaction['amount'] = float(tx['amount'].get('amount', 0))
                    transaction['currency'] = tx['amount'].get('currency', 'USD')
                else:
                    transaction['amount'] = float(tx['amount'])

            # Update debtor information
            if 'debtor' in pmt_info:
                transaction['debtor_name'] = pmt_info['debtor'].get('name')
                transaction['debtor_country'] = pmt_info['debtor'].get('country', 'UNKNOWN')

            if 'debtor_account' in pmt_info:
                transaction['debtor_account'] = pmt_info['debtor_account'].get('iban')

            # Update creditor information
            if 'creditor' in tx:
                transaction['creditor_name'] = tx['creditor'].get('name')
                transaction['creditor_country'] = tx['creditor'].get('country', 'UNKNOWN')

            if 'creditor_account' in tx:
                transaction['creditor_account'] = tx['creditor_account'].get('iban')

            # Update additional fields
            transaction['purpose_code'] = tx.get('purpose')
            transaction['remittance_info'] = tx.get('remittance_info')

            # Set cross-border flag
            transaction['is_cross_border'] = (
                transaction.get('debtor_country') != transaction.get('creditor_country')
            )

            # Set high-risk country flag
            high_risk_countries = {'AF', 'KP', 'IR', 'MM', 'SS', 'SY', 'VE', 'YE'}
            transaction['involves_high_risk_country'] = any(
                country in high_risk_countries
                for country in [transaction.get('debtor_country'), transaction.get('creditor_country')]
                if country
            )

        except Exception as e:
            logger.error(f"Error updating PAIN.001 transaction: {str(e)}")
            raise

    def _update_transaction_pacs008(self, transaction: Dict, tx: Dict) -> None:
        """Update transaction with PACS.008 specific information"""
        try:
            # Update amount and currency
            transaction['amount'] = float(tx.get('amount', 0))
            transaction['currency'] = tx.get('currency', 'USD')

            # Update debtor information
            transaction['debtor_name'] = tx.get('debtor_name')
            transaction['debtor_account'] = tx.get('debtor_account')
            
            # Try to extract country from BIC/IBAN
            debtor_country = self._extract_country_from_id(tx.get('debtor_account', ''))
            transaction['debtor_country'] = debtor_country or 'UNKNOWN'

            # Update creditor information
            transaction['creditor_name'] = tx.get('creditor_name')
            transaction['creditor_account'] = tx.get('creditor_account')
            
            # Try to extract country from BIC/IBAN
            creditor_country = self._extract_country_from_id(tx.get('creditor_account', ''))
            transaction['creditor_country'] = creditor_country or 'UNKNOWN'

            # Update additional fields
            transaction['purpose_code'] = tx.get('purpose')
            transaction['remittance_info'] = tx.get('remittance_info')

            # Set cross-border flag
            transaction['is_cross_border'] = (
                transaction['debtor_country'] != transaction['creditor_country']
            )

            # Set high-risk country flag
            high_risk_countries = {'AF', 'KP', 'IR', 'MM', 'SS', 'SY', 'VE', 'YE'}
            transaction['involves_high_risk_country'] = any(
                country in high_risk_countries
                for country in [transaction['debtor_country'], transaction['creditor_country']]
            )

        except Exception as e:
            logger.error(f"Error updating PACS.008 transaction: {str(e)}")
            raise

    def _update_transaction_camt054(self, transaction: Dict, notification: Dict, entry: Dict) -> None:
        """Update transaction with CAMT.054 specific information"""
        try:
            # Update amount and currency
            transaction['amount'] = float(entry.get('amount', 0))
            transaction['currency'] = entry.get('currency', 'USD')

            # Update account information
            if 'account' in notification:
                transaction['debtor_account'] = notification['account'].get('iban')
                # Try to extract country from IBAN
                debtor_country = self.parser._extract_country_from_id(transaction['debtor_account'] or '')
                transaction['debtor_country'] = debtor_country or 'UNKNOWN'

            # Update status and indicators
            transaction['credit_debit_indicator'] = entry.get('credit_debit')
            transaction['status'] = entry.get('status')

            # Parse and set booking and value dates using parser's _parse_datetime
            transaction['booking_date'] = self.parser._parse_datetime(entry.get('booking_date'))
            transaction['value_date'] = self.parser._parse_datetime(entry.get('value_date'))

            # Set flags based on available information
            transaction['is_cross_border'] = False  # Default for CAMT.054
            transaction['involves_high_risk_country'] = (
                transaction['debtor_country'] in {'AF', 'KP', 'IR', 'MM', 'SS', 'SY', 'VE', 'YE'}
            )

        except Exception as e:
            logger.error(f"Error updating CAMT.054 transaction: {str(e)}")
            raise

    def _extract_country_from_id(self, identifier: str) -> Optional[str]:
        """Extract country code from BIC or IBAN"""
        if not identifier:
            return None
            
        # Try IBAN first (first two characters)
        if len(identifier) >= 2 and identifier[:2].isalpha():
            return identifier[:2].upper()
            
        # Try BIC (characters 5-6)
        if len(identifier) >= 6:
            country_code = identifier[4:6]
            if country_code.isalpha():
                return country_code.upper()
                
        return None

    def _convert_to_transaction_format(self, parsed_message: Dict) -> Dict:
        

        """Convert parsed message to transaction format with comprehensive error handling"""
        try:
            # Initialize with safe defaults
            transaction = {
                'msg_id': '',
                'creation_date': datetime.now(),
                'channel': 'UNKNOWN',
                'amount': 0.0,
                'currency': 'USD',
                'debtor_account': '',
                'debtor_account_routing_number': '',
                'debtor_agent_BIC': '',
                'creditor_account': '',
                'creditor_account_routing_number': '',
                'creditor_name': '',
                'credit_agent_BIC': '',
                'purpose_code': '',
                'remittance_info': '',
                'mandate_id': '',
                'is_cross_border': False,
                'involves_high_risk_country': False,
                'booking_date': None,
                'value_date': None,
                'status': ''
            }

            # Safely get header information
            if parsed_message is not None:
                header = parsed_message.get('header')
                content = parsed_message.get('content', {})
                
                if header is not None:
                    transaction['msg_id'] = self.safe_strip(str(header.message_id))
                    transaction['creation_date'] = header.creation_datetime or datetime.now()
                    transaction['channel'] = self.safe_strip(
                        parsed_message.get('payment_channel', 'UNKNOWN').value
                        if hasattr(parsed_message.get('payment_channel'), 'value')
                        else 'UNKNOWN'
                    )

                # Process based on message type
                if header and header.message_type == ISO20022MessageType.PAIN_001:
                    if content.get('payment_informations'):
                        pmt_info = content['payment_informations'][0]
                        if pmt_info.get('transactions'):
                            tx = pmt_info['transactions'][0]
                            self._update_transaction_pain001(transaction, pmt_info, tx)
                
                elif header and header.message_type == ISO20022MessageType.PACS_008:
                    if content.get('credit_transfers'):
                        tx = content['credit_transfers'][0]
                        self._update_transaction_pacs008(transaction, tx)
                
                elif header and header.message_type == ISO20022MessageType.CAMT_054:
                    if content.get('notifications'):
                        notification = content['notifications'][0]
                        if notification.get('entries'):
                            entry = notification['entries'][0]
                            self._update_transaction_camt054(transaction, notification, entry)

            return transaction

        except Exception as e:
            logger.error(f"Error converting transaction format: {str(e)}")
            # Return safe defaults if conversion fails
            return transaction

    def _extract_country_from_bic(self, bic_or_iban: Optional[str]) -> Optional[str]:
        """Extract country code from BIC or IBAN"""
        if not bic_or_iban:
            return None
            
        # Try to extract from IBAN (first two characters)
        if len(bic_or_iban) >= 2:
            country_code = bic_or_iban[:2].upper()
            if country_code.isalpha():
                return country_code
                
        # Try to extract from BIC (characters 5-6)
        if len(bic_or_iban) >= 6:
            country_code = bic_or_iban[4:6].upper()
            if country_code.isalpha():
                return country_code
                
        return None
    
    

    def calculate_consolidated_risk(self, features,model_scores=None):
        """Calculate consolidated risk score and analysis. Delegates to detector."""
        try:
            # Delegate to detector's calculate_consolidated_risk method
            return self.detector.calculate_consolidated_risk(features)
        except Exception as e:
            logger.error(f"Error calculating consolidated risk: {str(e)}")
            # Return safe default values
            return {
                'consolidated_score': 0.0,
                'risk_level': 'LOW',
                'fraud_types': [],
                'risk_components': {},
                'contributing_factors': [],
                'base_score': 0.0,
                'high_risk_penalty': 0.0,
                'ato_indicators_triggered': 0
                # 'mule_indicators_triggered': 0
            }

    def _load_risk_weights(self) -> Dict:
        """Load risk weights from config"""
        try:
            return {
                'ml_score': float(self.config_reader.get_property('risk.weightage_check.ml_score', '0.25')),
                'rule_based': float(self.config_reader.get_property('risk.weightage_check.rule_based', '0.25')),
                'pattern_based': float(self.config_reader.get_property('risk.weightage_check.pattern_based', '0.15')),
                'channel_specific': float(self.config_reader.get_property('risk.weightage_check.channel_specific', '0.15')),
                'app_fraud': float(self.config_reader.get_property('risk.weightage_check.app_fraud', '0.20')),  # New weight
                'account_takeover':float(self.config_reader.get_property('risk.weightage_check.account_takeover', '0.20'))
                # 'mule_account':float(self.config_reader.get_property('risk.weightage_check.mule_account','0.1'))        
        
            }
        except Exception as e:
            logger.error(f"Error loading risk weights from config: {str(e)}")
            # Return defaults if there's an error
            return {
                'ml_score': 0.3,
                'rule_based': 0.3,
                'pattern_based': 0.2,
                'channel_specific': 0.2
            }
    def _check_cross_border(self, debtor_country: Optional[str], creditor_country: Optional[str]) -> bool:
        """Check if transaction is cross-border"""
        if debtor_country and creditor_country:
            return debtor_country != creditor_country
        return False

    def _check_high_risk_countries(self, debtor_country: Optional[str], creditor_country: Optional[str]) -> bool:
        """Check if any country is in high-risk list"""
        high_risk_countries = {'AF', 'KP', 'IR', 'MM', 'SS', 'SY', 'VE', 'YE'}
        return any(
            country in high_risk_countries
            for country in [debtor_country, creditor_country]
            if country is not None
        )

    def _perform_business_entity_checks(self, transaction: Dict) -> Dict:
        """
        Perform business entity and NAICS checks
        """
        try:
            # Check debtor name
            debtor_check = self.db.check_business_entity(
                transaction.get('debtor_name', '')
            )
            
            # Check creditor name
            creditor_check = self.db.check_business_entity(
                transaction.get('creditor_name', '')
            )
            
            # Determine business entity status
            is_business = debtor_check['is_business'] or creditor_check['is_business']
            
            # Collect NAICS codes
            naics_codes = (
                debtor_check.get('naics_codes', []) + 
                creditor_check.get('naics_codes', [])
            )
            
            # Check NAICS risk
            naics_risk = self.naics_risk_checker.check_naics_risk(naics_codes)
            
            return {
                'is_business': is_business,
                'debtor_business_details': debtor_check,
                'creditor_business_details': creditor_check,
                'risk_score':naics_risk['risk_score'],
                'naics_risk': naics_risk
            }
        
        except Exception as e:
            logger.error(f"Error performing business entity checks: {str(e)}")
            # Return default result if checks fail
            return {
                'is_business': False,
                'debtor_business_details': {'is_business': False},
                'creditor_business_details': {'is_business': False},
                'naics_risk': {
                    'is_risky': False,
                    'risk_score': 0,
                    'matched_naics_codes': []
                }
            }

    def consolidate_alerts(alerts: List[str]) -> str:
        """
        Consolidate multiple alerts into a single, readable description
        
        :param alerts: List of individual alert messages
        :return: Consolidated alert description
        """
        # Remove duplicates while preserving order
        unique_alerts = list(dict.fromkeys(alerts))
        
        # Categorize and prioritize alerts
        categories = {
            'channel': [],
            'geographic': [],
            'timing': [],
            'other': []
        }
        
        for alert in unique_alerts:
            alert = alert.strip()
            if not alert:
                continue
            
            if 'payment channel' in alert.lower() or 'channel' in alert.lower():
                categories['channel'].append(alert)
            elif any(term in alert.lower() for term in ['country', 'cross-border', 'high-risk']):
                categories['geographic'].append(alert)
            elif any(term in alert.lower() for term in ['hours', 'time', 'business']):
                categories['timing'].append(alert)
            else:
                categories['other'].append(alert)
        
        # Construct consolidated description
        description_parts = []
        
        # Channel alerts
        if categories['channel']:
            description_parts.append(". ".join(categories['channel']))
        
        # Geographic alerts
        if categories['geographic']:
            description_parts.append(". ".join(categories['geographic']))
        
        # Timing alerts
        if categories['timing']:
            description_parts.append(". ".join(categories['timing']))
        
        # Other alerts
        if categories['other']:
            description_parts.append(". ".join(categories['other']))
        
        # Combine all parts
        consolidated_description = ". ".join(description_parts)
        
        return consolidated_description.capitalize() + "."
    
    def _assess_swift_risk(self, transaction: Dict, metrics: TransactionMetrics) -> float:
        """
        Perform a specific risk assessment for SWIFT transactions
        
        :param transaction: Transaction details
        :param metrics: Transaction metrics
        :return: Risk score (0-100)
        """
        try:
            swift_risk_score = 0.0
            
            # Check cross-border transfer
            if transaction.get('is_cross_border', False):
                swift_risk_score += 30
            
            # Check high-risk country involvement
            if transaction.get('involves_high_risk_country', False):
                swift_risk_score += 40
            
            # Check transaction amount
            amount = float(transaction.get('amount', 0))
            amount_deviation = abs(amount - metrics.avg_amount_30d) / (metrics.std_amount_30d if metrics.std_amount_30d > 0 else 1)
            
            if amount_deviation > 2:  # More than 2 standard deviations
                swift_risk_score += 20
            
            # Check unique transaction characteristics
            if metrics.new_recipient:
                swift_risk_score += 10
            
            return min(swift_risk_score, 100)
        
        except Exception as e:
            logger.error(f"Error in SWIFT risk assessment: {str(e)}")
            return 50  # Default moderate risk

    def _check_for_anomalies(self, transaction: Dict, metrics: TransactionMetrics, history_df: pd.DataFrame) -> bool:
        """
        Detect anomalies in transaction patterns compared to historical behavior
        """
        # Initialize the result dictionary
        anomaly_result = {
            'is_anomalous': False,
            'detected_anomalies': [],
            'risk_score_increment': 0
        }
        
        # Early return if not enough history
        if history_df.empty or len(history_df) < 5:
            # Not enough history to determine patterns
            return anomaly_result
        
        try:
            # Initialize timezone manager if needed
            if not hasattr(self, 'timezone_manager'):
                self.timezone_manager = TimeZoneManager()
            
            # Get basic transaction information
            country_code = transaction.get('debtor_country', 'UNKNOWN')
            payment_channel = transaction.get('channel', 'UNKNOWN')
            amount = float(transaction.get('amount', 0))
            recipient = transaction.get('creditor_account', '')
            transaction_time = pd.to_datetime(transaction.get('creation_date'))
            
            # Get local time for proper comparison
            if country_code != 'UNKNOWN':
                local_time = self.timezone_manager.convert_to_local_time(transaction_time, country_code)
                current_hour = local_time.hour
                
                # Convert history times to local time for fair comparison
                if not history_df.empty:
                    history_df['local_hour'] = history_df['creation_date'].apply(
                        lambda dt: self.timezone_manager.convert_to_local_time(dt, country_code).hour
                    )
                    usual_hours = history_df['local_hour'].value_counts()
                else:
                    usual_hours = pd.Series()
            else:
                # Fallback to server time
                current_hour = transaction_time.hour
                if not history_df.empty:
                    usual_hours = history_df['creation_date'].dt.hour.value_counts()
                else:
                    usual_hours = pd.Series()
            
            # Filter history by payment channel for more accurate comparisons
            channel_history = history_df[history_df['channel'] == payment_channel] if not history_df.empty else pd.DataFrame()
            
            # If no channel history, fall back to all history
            if channel_history.empty or len(channel_history) < 3:
                channel_history = history_df
            
            # Get channel-specific thresholds from config
            amount_zscore_threshold = float(self.config_reader.get_property(
                f'{payment_channel.lower()}.anomaly.amount_zscore_threshold', 
                self.config_reader.get_property('anomaly.amount_zscore_threshold', '2.5')
            ))
            
            velocity_multiplier = float(self.config_reader.get_property(
                f'{payment_channel.lower()}.anomaly.velocity_multiplier', 
                self.config_reader.get_property('anomaly.velocity_multiplier', '2.0')
            ))
            
            high_amount_multiplier = float(self.config_reader.get_property(
                f'{payment_channel.lower()}.anomaly.high_amount_multiplier', '1.5'
            ))
            
            # --------- ANOMALY DETECTION CHECKS ---------
            
            # 1. Time pattern anomaly - unusual hour for this account
            if not usual_hours.empty and current_hour not in usual_hours.index[:3] and len(usual_hours) >= 3:
                anomaly_detail = f"Unusual time pattern: Transaction at hour {current_hour} local time ({country_code}) is uncommon for this account using {payment_channel}"
                logger.info(f"Anomaly detected: {anomaly_detail}")
                anomaly_result['detected_anomalies'].append({
                    'type': 'UNUSUAL_TIME_PATTERN',
                    'severity': 'MEDIUM', 
                    'details': anomaly_detail,
                    'risk_increment': 15
                })
                anomaly_result['is_anomalous'] = True
                anomaly_result['risk_score_increment'] += 15
            
            # 2. Amount anomaly - transaction amount significantly different from history
            if metrics.std_amount_30d > 0:
                z_score = abs((amount - metrics.avg_amount_30d) / metrics.std_amount_30d)
                if z_score > amount_zscore_threshold:
                    anomaly_detail = f"Unusual amount: ${amount:.2f} deviates significantly from normal patterns (z-score: {z_score:.2f})"
                    logger.info(f"Anomaly detected: {anomaly_detail}")
                    
                    # Higher risk increment for very large deviations
                    risk_increment = 20
                    if z_score > amount_zscore_threshold * 2:
                        risk_increment = 35  # Much higher risk for extreme deviation
                    
                    anomaly_result['detected_anomalies'].append({
                        'type': 'UNUSUAL_AMOUNT',
                        'severity': 'HIGH' if z_score > amount_zscore_threshold * 2 else 'MEDIUM',
                        'details': anomaly_detail,
                        'risk_increment': risk_increment,
                        'z_score': z_score
                    })
                    anomaly_result['is_anomalous'] = True
                    anomaly_result['risk_score_increment'] += risk_increment
            
            # 3. Recipient anomaly - new recipient with unusually high amount
            if metrics.new_recipient and amount > metrics.avg_amount_30d * high_amount_multiplier:
                anomaly_detail = f"New recipient with unusually high amount: ${amount:.2f} is {high_amount_multiplier}x higher than average"
                logger.info(f"Anomaly detected: {anomaly_detail}")
                anomaly_result['detected_anomalies'].append({
                    'type': 'NEW_RECIPIENT_HIGH_AMOUNT',
                    'severity': 'HIGH',
                    'details': anomaly_detail,
                    'risk_increment': 25
                })
                anomaly_result['is_anomalous'] = True
                anomaly_result['risk_score_increment'] += 25
            
            # 4. Velocity anomaly - sudden increase in transaction frequency
            recent_velocity = metrics.velocity_24h
            avg_velocity = len(channel_history) / 30  # Average daily transactions over last 30 days
            if recent_velocity > avg_velocity * velocity_multiplier:
                anomaly_detail = f"Unusual transaction frequency: {recent_velocity} transactions in 24h is {recent_velocity/(avg_velocity or 1):.1f}x the normal rate"
                logger.info(f"Anomaly detected: {anomaly_detail}")
                anomaly_result['detected_anomalies'].append({
                    'type': 'VELOCITY_SPIKE',
                    'severity': 'MEDIUM',
                    'details': anomaly_detail,
                    'risk_increment': 20
                })
                anomaly_result['is_anomalous'] = True
                anomaly_result['risk_score_increment'] += 20
            
            # 5. Channel-specific anomaly checks
            if payment_channel == 'ZELLE':
                # Special check for Zelle: multiple new recipients in short time
                if metrics.unique_recipients_24h > float(self.config_reader.get_property('zelle.anomaly.max_new_recipients', '3')):
                    anomaly_detail = f"Multiple new Zelle recipients: {metrics.unique_recipients_24h} different recipients in 24h"
                    logger.info(f"Anomaly detected: {anomaly_detail}")
                    anomaly_result['detected_anomalies'].append({
                        'type': 'ZELLE_MULTIPLE_RECIPIENTS',
                        'severity': 'HIGH',
                        'details': anomaly_detail,
                        'risk_increment': 30
                    })
                    anomaly_result['is_anomalous'] = True
                    anomaly_result['risk_score_increment'] += 30
            elif payment_channel == 'WIRE':
                # Special check for Wire: international wire outside business hours
                is_cross_border = transaction.get('is_cross_border', False)
                is_business_hours = (9 <= current_hour <= 17) and (local_time.weekday() < 5)
                
                if is_cross_border and not is_business_hours:
                    anomaly_detail = f"International wire outside business hours"
                    logger.info(f"Anomaly detected: {anomaly_detail}")
                    anomaly_result['detected_anomalies'].append({
                        'type': 'WIRE_UNUSUAL_TIMING',
                        'severity': 'MEDIUM',
                        'details': anomaly_detail,
                        'risk_increment': 20
                    })
                    anomaly_result['is_anomalous'] = True
                    anomaly_result['risk_score_increment'] += 20
            elif payment_channel == 'ACH':
                # Special check for ACH: same day with high value
                same_day_threshold = float(self.config_reader.get_property('ach.anomaly.same_day_threshold', '5000'))
                if transaction.get('same_day', False) and amount > same_day_threshold:
                    anomaly_detail = f"High-value same-day ACH: ${amount:.2f}"
                    logger.info(f"Anomaly detected: {anomaly_detail}")
                    anomaly_result['detected_anomalies'].append({
                        'type': 'ACH_HIGH_VALUE_SAME_DAY',
                        'severity': 'HIGH',
                        'details': anomaly_detail,
                        'risk_increment': 25
                    })
                    anomaly_result['is_anomalous'] = True
                    anomaly_result['risk_score_increment'] += 25
            
            # Cap the total risk score increment
            max_risk_increment = float(self.config_reader.get_property('anomaly.max_risk_increment', '50'))
            anomaly_result['risk_score_increment'] = min(anomaly_result['risk_score_increment'], max_risk_increment)
            
            return anomaly_result
                
        except Exception as e:
            logger.error(f"Error in anomaly detection: {str(e)}")
            return anomaly_result  # Return the result object instead of False
        
    
    # ---Changing the process --------#
    # -- Restructured the method to implement the flow as described: first check ML confidence, then either take a fast path #
    # -- (for high confidence) or process all detectors (for low confidence), and always end with database operations.#
    # --------------------------------# 

    def _process_fraud_detection(self, transaction: Dict, result: Dict) -> Dict:
        """Process fraud detection with error handling"""

        #Step A: Business Entity Checks
        #Raise the risk score if the entity activity risk is configured, e.g. Casinos increase risk profile 
        try:
            # Reset relevant scores and alerts for business checks
            result['final_risk_score'] = 0
            result['compliance_score'] = 0
            result['alerts'] = []
            result['risk_factors']['compliance_alerts'] = []
            result['compliance_risk_score'] = 0

            # Perform business entity checks
            be_starttime = time.time()
            business_checks = self._perform_business_entity_checks(transaction)
            transaction['business_entity_check'] = business_checks

            # Add business risk to final score if applicable
            if business_checks.get('risk_score'):
                result['final_risk_score'] += float(business_checks['risk_score'])

            # Add business information to risk factors
            result['risk_factors'].update({
                'is_business': business_checks.get('is_business', False),
                'business_risk_score': business_checks.get('risk_score', 0),
                'naics_code': business_checks.get('naics_risk', {}).get('matched_naics_codes', []),
                'business_type': business_checks.get('business_type', 'UNKNOWN')
            })
            
            # Add business-related alerts if high risk
            naics_risk_score = business_checks.get('risk_score', 0)
            naics_threshold = float(self.config_reader.get_property('business.naics.risk.threshold', '70'))
            
            if naics_risk_score > naics_threshold:
                alert = {
                    'type': 'HIGH_RISK_BUSINESS_TYPE',
                    'severity': 'HIGH',
                    'reason': f"High-risk business type detected (NAICS risk score: {naics_risk_score})",
                    'details': {
                        'naics_codes': business_checks.get('naics_risk', {}).get('matched_naics_codes', []),
                        'risk_score': naics_risk_score
                    }
                }
                result['alerts'].append(alert['reason'])
                result['risk_factors']['compliance_alerts'].append(alert)
                
        except Exception as e:
            self.logger.error(f"Error in business entity checks: {str(e)}")
            # Set default values in case of failure
            transaction['business_entity_check'] = {
                'is_business': False,
                'risk_score': 0,
                'naics_risk': {'risk_score': 0, 'matched_naics_codes': []}
            }

        be_endtime = time.time()
        elapsed_be = be_endtime-be_starttime
        # print(f"Elapsed time in _perform_business_entity_checks - {elapsed_be:.6f} seconds")
            
        be_starttime = time.time()
        try:
            ach_return_rate_check = self._check_ach_return_rates(transaction)
            
            # Add ACH return rate alerts if applicable
            if ach_return_rate_check.get('alerts'):
                # result['is_suspicious'] = True
                
                # Add alerts to risk factors
                result['risk_factors']['ach_return_rate_alerts'] = ach_return_rate_check.get('alerts', [])
                
                # Update risk score
                if ach_return_rate_check.get('is_high_risk'):
                    result['final_risk_score'] += float(
                        self.config_reader.get_property('ach.return_rate.risk_score_increment', '20')
                    )
                
                # Add alert reasons to main alerts list
                for alert in ach_return_rate_check.get('alerts', []):
                    if isinstance(alert, dict):
                        result['alerts'].append(alert.get('details', ''))
                    elif alert:
                        result['alerts'].append(str(alert))
            
            # Add return rate metrics to transaction details
            transaction['ach_return_rate_metrics'] = ach_return_rate_check.get('return_rate_metrics', {})

        except Exception as e:
            logger.error(f"Error performing ACH return rate checks: {str(e)}")

        be_endtime = time.time()
        elapsed_be = be_endtime-be_starttime
        # print(f"Elapsed time in _check_ach_return_rates - {elapsed_be:.6f} seconds")
        
        
        try:
            # Define weights 
            weights = self._load_risk_weights()

            # Load transaction history
            be_starttime = time.time()
            history_df = pd.DataFrame()
            if transaction.get('debtor_account'):
                history_loader = TransactionHistoryLoader(
                    db_connection=self.db,
                    config_reader=self.config_reader,
                    payment_channel=transaction.get('channel', 'UNKNOWN')
                )
                history_df = history_loader.load_account_history(
                    transaction['debtor_account'],
                    lookback_days=int(self.config_reader.get_property('transaction.history.lookback.days', '30'))
                )

            be_endtime = time.time()
            elapsed_be = be_endtime - be_starttime
            # print(f"Elapsed time in load history {elapsed_be:.6f} seconds")
            # Create default metrics
            default_metrics = TransactionMetrics(
                velocity_24h=0,
                amount_24h=0.0,
                unique_recipients_24h=0,
                velocity_7d=0,
                amount_7d=0.0,
                unique_recipients_7d=0,
                avg_amount_30d=float(transaction.get('amount', 0.0)),
                std_amount_30d=0.0,
                new_recipient=True,
                cross_border=transaction.get('is_cross_border', False),
                high_risk_country=transaction.get('involves_high_risk_country', False)
            )

            # Compute metrics
            be_starttime = time.time()
            try:
                metrics = self.analyzer.compute_metrics(transaction, history_df)
                if metrics is None:
                    metrics = default_metrics
            except Exception as e:
                logger.error(f"Error computing metrics: {str(e)}")
                metrics = default_metrics

            
            be_endtime = time.time()
            elapsed_be = be_endtime - be_starttime
            # print(f"Elapsed time in compute metrics {elapsed_be:.6f} seconds")
            # Extract features
            try:
                features = self.feature_extractor.extract_features(transaction, metrics)
            except Exception as e:
                logger.error(f"Error extracting features: {str(e)}")
                return {'error': True, 'result': result}

            # Get ensemble prediction
            ml_score = self.get_ensemble_prediction(features)

            # Extract confidence score from ML prediction
            ml_confidence = ml_score.get('confidence', 0) if isinstance(ml_score, dict) else 0

            # Define threshold for high confidence
            HIGH_CONFIDENCE_THRESHOLD = float(self.config_reader.get_property('ml.high_confidence.threshold', '0.75'))

            # Process ML results
            ml_score_value = 0
            if isinstance(ml_score, dict):
                ml_score_value = ml_score.get('score', 0)
                result['ml_score'] = ml_score_value
                result['ml_confidence'] = ml_confidence
                result['ml_fraud_types'] = ml_score.get('fraud_types', [])
                
                # Add any fraud types detected by ML model to the main list
                if 'fraud_types' not in result:
                    result['fraud_types'] = []
                result['fraud_types'].extend(ml_score.get('fraud_types', []))
            else:
                ml_score_value = ml_score
                result['ml_score'] = ml_score_value
                result['ml_confidence'] = ml_confidence

            # Scale ML score to 0-100 range for consistency with other scores
            ml_score_scaled = ml_score_value * 100
            
            # DECISION POINT: Based on ML confidence, take either fast path or full analysis
            if ml_confidence >= HIGH_CONFIDENCE_THRESHOLD:
                # logger.info(f"Taking fast path with ML confidence: {ml_confidence} and score is {ml_score_scaled}")
                
                # Step 1: Add existing risk score to ML-based score
                initial_risk_score = result.get('final_risk_score', 0)
                ml_weight = weights.get('ml_score', 0.25)
                
                # Apply ML score with higher weight since we're skipping other checks
                fast_path_ml_weight = float(self.config_reader.get_property('ml.high_confidence.weight', '0.75'))
                result['final_risk_score'] = initial_risk_score + (ml_score_scaled * fast_path_ml_weight)
                
                # Set default empty values for skipped detectors
                result['risk_factors']['app_fraud_summary'] = {
                    'is_suspected_app_fraud': False,
                    'detection_score': 0,
                    'weights': 0,
                    'risk_score_calculated': 0,
                    'triggered_indicators': []
                }
                result['risk_factors']['app_fraud_indicators'] = []
                
                result['risk_factors']['mule_detection_summary'] = {
                    'is_suspected_mule': False,
                    'detection_score': 0,
                    'weights': 0,
                    'risk_score_calculated': 0,
                    'triggered_indicators': []
                }
                result['risk_factors']['mule_detection_indicators'] = []
                
                # Record that we took the fast path
                result['processing_path'] = 'FAST_PATH'
                result['processing_reason'] = f"High ML confidence ({ml_confidence:.2f})"
                
                # Add ML reason to risk factors
                if ml_score_scaled > 60:  # Adjust threshold as needed
                    result['alerts'].append(f"ML model detected high risk with confidence {ml_confidence:.2f}")
                
            else:
                # logger.info(f"Taking full analysis path with ML confidence: {ml_confidence}")
                result['processing_path'] = 'FULL_ANALYSIS'
                result['processing_reason'] = f"Low ML confidence ({ml_confidence:.2f})"
                
                # Get channel-specific alerts
                be_starttime = time.time()
                try:
                    channel_alerts = self.channel_detector.detect_unusual_patterns(transaction, metrics)
                    channel_score, channel_factors = self.channel_detector.calculate_alert_risk_score(channel_alerts)
                except Exception as e:
                    logger.error(f"Error in channel detection: {str(e)}")
                    channel_alerts = []
                    channel_score = 0
                    channel_factors = []
                    
                be_endtime = time.time()
                elapsed_be = be_endtime - be_starttime
                # print(f"Elapsed time in detect_unusual_patterns {elapsed_be:.6f} seconds")

                # Update result with channel alerts
                if channel_alerts:
                    result['alerts'].extend(alert.get('details', '') for alert in channel_alerts)
                    result['risk_factors']['channel_alerts'] = channel_factors

                # First, run anomaly detection for all fraud types for sudden changes in behavior
                anomaly_result = None
                if transaction.get('debtor_account'):
                    be_starttime = time.time()
                    try:
                        # Check if transaction is a notable deviation from previous patterns
                        anomaly_result = self._check_for_anomalies(transaction, metrics, history_df)
                    except Exception as e:
                        logger.error(f"Error in anomaly detection: {str(e)}")
                    be_endtime = time.time()
                    elapsed_be = be_endtime - be_starttime
                    # print(f"Elapsed time in _check_for_anomalies {elapsed_be:.6f} seconds")

                # Run APP fraud detection
                try:
                    if 'risk_factors' not in result:
                        result['risk_factors'] = {}
        
                    if 'app_fraud_indicators' not in result['risk_factors']:
                        result['risk_factors']['app_Fraud_indicators'] = {}
                        
                    be_starttime = time.time()
                    app_fraud_results = self.app_detector.check_transaction_for_app_fraud(
                        transaction,
                        history_df,
                        anomaly_result
                    )
                    be_endtime = time.time()
                    elapsed_be= be_endtime - be_starttime
                    # print(f"Elapsed time in check_transaction_for_app_fraud {elapsed_be:.6f} seconds")


                    if not app_fraud_results or not app_fraud_results.get('is_suspected_app_fraud', False): 
                        # Create default empty result
                        result['risk_factors']['app_fraud_summary'] = {
                            'is_suspected_app_fraud': False,
                            'detection_score': 0,
                            'weights': 0,
                            'risk_score_calculated': 0,
                            'triggered_indicators': []
                        }
                        result['risk_factors']['app_fraud_indicators'] = []

                    # Process APP fraud results
                    if app_fraud_results.get('is_suspected_app_fraud', False):
                        # Update risk factors
                        result['risk_factors']['app_fraud_indicators'] = app_fraud_results.get('behavioral_indicators', [])
                        result['risk_factors']['app_fraud_confidence'] = app_fraud_results.get('confidence_level', 'LOW')
                        
                        result['final_risk_score'] += app_fraud_results.get('risk_score', 0) * weights.get('app_fraud', 0.2)
                        
                        # Add alerts
                        for factor in app_fraud_results.get('contributing_factors', []):
                            result['alerts'].append(f"APP FRAUD: {factor}")
                        
                        # Add fraud type
                        if 'fraud_types' not in result:
                            result['fraud_types'] = []
                        result['fraud_types'].append('SUSPECTED_APP_FRAUD')
                except Exception as e:
                    logger.error(f"Error in APP fraud detection: {str(e)}")

                # # Run mule account detection
                # be_starttime = time.time()
                # try:
                #     mule_detection_results = self.mule_detector.detect_mule_patterns(
                #         transaction, history_df
                #     )
                    
                #     if not mule_detection_results or not mule_detection_results.get('is_suspected_mule', False): 
                #         # Create default empty result
                #         result['risk_factors']['mule_detection_summary'] = {
                #             'is_suspected_mule': False,
                #             'detection_score': 0,
                #             'weights': 0,
                #             'risk_score_calculated': 0,
                #             'triggered_indicators': []
                #         }
                #         result['risk_factors']['mule_detection_indicators'] = []
                        
                #     # Process results and update risk factors
                #     if mule_detection_results and mule_detection_results['is_suspected_mule']:
                #         result['final_risk_score'] += mule_detection_results['risk_score'] * weights.get('mule_account', 0.2)
                        
                #         # Add alerts
                #         for indicator in mule_detection_results['triggered_indicators']:
                #             result['alerts'].append(f"MULE ACCOUNT: {indicator['details']}")
                        
                #         # Add fraud type
                #         if 'fraud_types' not in result:
                #             result['fraud_types'] = []
                #         result['fraud_types'].append('SUSPECTED_MULE_ACCOUNT')
                # except Exception as e:
                #     logger.error(f"Error in mule account detection: {str(e)}")

                # be_endtime = time.time()
                # elapsed_be= be_endtime - be_starttime
                # print(f"Elapsed time in detect_mule_patterns {elapsed_be:.6f} seconds")

                # be_starttime = time.time()
                # # Run mule relationship detection when transferring funds
                # try:
                #     mule_detection_threshold = float(self.config_reader.get_property('mule_detection_threshold', '5000'))
                #     if transaction.get('amount', 0) > mule_detection_threshold:
                #         # Get sender and recipient account IDs
                #         sender_id = transaction.get('debtor_account')
                #         recipient_id = transaction.get('creditor_account')
                        
                #         # Skip if either ID is missing
                #         if sender_id and recipient_id:
                #             # Check for mule patterns
                #             mule_relationship_results = self.mule_detector.detect_mule_relationships(
                #                 sender_id=sender_id,
                #                 recipient_id=recipient_id,
                #                 transaction_amount=transaction.get('amount', 0),
                #                 transaction_time=transaction.get('timestamp')
                #             )
                            
                #             # Add mule score to the risk factors
                #             if mule_relationship_results.get('is_suspicious', False):
                #                 # Add to final risk score
                #                 result['final_risk_score'] += mule_relationship_results.get('risk_score', 0) * weights.get('mule_relationship', 0.25)
                                
                #                 # Add alerts for suspicious patterns
                #                 for pattern in mule_relationship_results.get('patterns_detected', []):
                #                     result['alerts'].append(f"MULE NETWORK: {pattern}")
                                    
                #                 # Add fraud type
                #                 if 'fraud_types' not in result:
                #                     result['fraud_types'] = []
                #                 result['fraud_types'].append('SUSPECTED_MULE_NETWORK')
                                
                #                 # Add to high priority alerts if very suspicious
                #                 mule_relationship_threshold = float(self.config_reader.get_property('mule.detection.mule_high_priority_threshold', '0.85'))
                #                 if mule_relationship_results.get('risk_score', 0) > mule_relationship_threshold:
                #                     if 'priority' not in result:
                #                         result['priority'] = 'HIGH'
                #                     result['priority_reason'] = f"Potential money mule network activity detected with confidence {mule_relationship_results.get('risk_score', 0):.2f}"
                # except Exception as e:
                #     logger.error(f"Error in mule relationship detection: {str(e)}")
                
                # be_endtime = time.time()
                # elapsed_be= be_endtime - be_starttime
                # print(f"Elapsed time in mule relationship detection {elapsed_be:.6f} seconds")


                # Run account takeover detection
                be_starttime = time.time()
                try:
                    if 'risk_factors' not in result:
                        result['risk_factors'] = {}

                    ato_detection_results = self.ato_detector.detect_account_takeover(
                        transaction, metrics, history_df
                    )

                    if not ato_detection_results or not ato_detection_results.get('is_suspected_takeover', False): 
                        # Create default empty result
                        ato_detection_results = {
                            'is_suspected_takeover': False,
                            'detection_score': 0,
                            'weights': 0,
                            'risk_score_calculated': 0,
                            'triggered_indicators': []
                        }
                    
                    # Process results and update risk factors
                    if ato_detection_results and ato_detection_results['is_suspected_takeover']:
                        result['final_risk_score'] += ato_detection_results['risk_score'] * weights.get('account_takeover', 0.2)
                        # Add alerts
                        for indicator in ato_detection_results['triggered_indicators']:
                            result['alerts'].append(f"Possible ATO Scenario: {indicator['details']}")
                        
                        # Add fraud type
                        if 'fraud_types' not in result:
                            result['fraud_types'] = []
                        result['fraud_types'].append('SUSPECTED_ACCOUNT_TAKEOVER')
                except Exception as e:
                    logger.error(f"Error in account takeover detection: {str(e)}")
                
                be_endtime = time.time()
                elapsed_be= be_endtime - be_starttime
                # print(f"Elapsed time in detect_account_takeover {elapsed_be:.6f} seconds")
                
                # Calculate consolidated risk

                try:
                    features_dict = self._prepare_features_dict(transaction, metrics)
                    risk_analysis = self.calculate_consolidated_risk(features_dict)
                except Exception as e:
                    logger.error(f"Error calculating consolidated risk: {str(e)}")
                    risk_analysis = {
                        'consolidated_score': 0,
                        'risk_components': {},
                        'contributing_factors': [],
                        'ato_indicators_triggered': 0,
                        'mule_indicators_triggered': 0
                    }

                # Calculate final risk score
                try:
                    consolidated_score = risk_analysis.get('consolidated_score', 0)
                    
                    calc_final_risk_score = result['final_risk_score'] + (
                        ml_score_scaled * weights.get('ml_score', 0.25) +
                        consolidated_score * weights.get('rule_based', 0.25) +
                        channel_score * weights.get('channel_specific', 0.15)
                        # APP fraud score, mule detection, etc. already added to result['final_risk_score']
                    )
                    
                    # Update result with computed values
                    self._improved_update_result_with_computed_values(
                        result=result, 
                        transaction=transaction,
                        metrics=metrics, 
                        ml_score=ml_score_scaled,
                        channel_score=channel_score, 
                        app_fraud_results=app_fraud_results if 'app_fraud_results' in locals() else None,
                        risk_analysis=risk_analysis, 
                        calc_final_risk_score=calc_final_risk_score
                    )
                    
                except Exception as e:
                    logger.error(f"Error calculating final risk score: {str(e)}")
                    return {'error': True, 'result': result}

            # Generate business explanations (for both fast and full paths)
            try:
                # Generate business-friendly explanations
                business_explanation = self.business_explainer.generate_business_explanation(transaction, result)
                result['business_explanation'] = business_explanation
                
                # Generate executive summary with consolidated alerts
                executive_summary = self.business_explainer.generate_executive_summary(transaction, result)
                result['executive_summary'] = executive_summary
            except Exception as e:
                logger.error(f"Error generating business explanations: {str(e)}")

            # # Process database operations (for both fast and full paths)
            # try:
            #     self.process_database_operations(transaction, result)
            # except Exception as e:
            #     logger.error(f"Error in database operations: {str(e)}")

            return {'error': False, 'result': result}

        except Exception as e:
            logger.error(f"Error in fraud detection: {str(e)}")
            return {'error': True, 'result': result}
       
    
    
    def _check_all_platform_lists(self, transaction: Dict) -> Dict:
        """
        Check all platform lists (whitelist, blacklist, watchlist) in a single operation.
        
        Returns a dict with results for each list type.
        """
        import time
        
        results = {
            'whitelist': {
                'is_whitelisted': False,
                'matched_entry': None,
                'match_type': None
            },
            'blacklist': {
                'is_blacklisted': False,
                'matched_entry': None,
                'match_type': None,
                'risk_score': 0,
                'risk_level': 'LOW',
                'severity': 'LOW'
            },
            'watchlist': {
                'is_watchlisted': False,
                'matched_entry': None,
                'match_type': None,
                'risk_score': 0,
                'risk_level': 'LOW'
            }
        }
        
        try:
            # Extract all relevant identifiers from the transaction
            identifiers = []
            
            # Add debtor identifiers
            if transaction.get('debtor_name'):
                identifiers.append(('entity_name', transaction['debtor_name'].lower(), 'debtor'))
            if transaction.get('debtor_account'):
                identifiers.append(('account_number', transaction['debtor_account'].lower(), 'debtor_bank'))
            if transaction.get('debtor_account_routing_number'):
                identifiers.append(('bank_routing_number', transaction['debtor_account_routing_number'].lower(), 'debtor_bank'))
            if transaction.get('debtor_country'):
                identifiers.append(('country', transaction['debtor_country'].lower(), 'debtor_country'))
                
            # Add creditor identifiers
            if transaction.get('creditor_name'):
                identifiers.append(('entity_name', transaction['creditor_name'].lower(), 'creditor'))
            if transaction.get('creditor_account_number'):
                identifiers.append(('account_number', transaction['creditor_account_number'].lower(), 'creditor_bank'))
            if transaction.get('creditor_account_routing_number'):
                identifiers.append(('bank_routing_number', transaction['creditor_account_routing_number'].lower(), 'creditor_bank'))
            if transaction.get('creditor_country'):
                identifiers.append(('country', transaction['creditor_country'].lower(), 'creditor_country'))
            
            # If no identifiers, return early
            if not identifiers:
                return results
                
            # Extract metadata
            debtor_country = transaction.get('debtor_country')
            creditor_country = transaction.get('creditor_country')
            channel = transaction.get('channel')
            
            # Prepare identifier values and types for a single query
            info_types = []
            info_values = []
            
            for info_type, info_value, _ in identifiers:
                info_types.append(info_type)
                info_values.append(info_value)
                
            # Make unique
            unique_info_types = list(set(info_types))
            unique_info_values = list(set(info_values))
            
            # Build query to check all lists at once
            self.db.reconnect()
            
            # START TIMING
            # query_start = time.time()
            
            query = """
            SELECT 
                i.list_type,
                i.information_type,
                i.information_value,
                e.*
            FROM 
                platform_list_items i
            JOIN 
                platform_list_entities e ON i.list_entity_id = e.list_entity_id
            WHERE 
                (
                    i.expiry_date IS NULL OR i.expiry_date >= CURRENT_DATE()
                )
                AND (
                    (e.start_time IS NULL AND e.end_time IS NULL) OR 
                    (e.start_time IS NULL AND e.end_time >= UTC_TIMESTAMP()) OR 
                    (e.start_time <= UTC_TIMESTAMP() AND e.end_time IS NULL) OR 
                    (e.start_time <= UTC_TIMESTAMP() AND e.end_time >= UTC_TIMESTAMP())
                )
            """
            
            params = []
            
            # Add info_types to query if any
            if unique_info_types:
                type_placeholders = ', '.join(['%s'] * len(unique_info_types))
                query += f" AND i.information_type IN ({type_placeholders})"
                params.extend(unique_info_types)
                
            # Add info_values to query
            if unique_info_values:
                value_placeholders = ', '.join(['%s'] * len(unique_info_values))
                query += f" AND LOWER(i.information_value) IN ({value_placeholders})"
                params.extend(unique_info_values)
                
            # Add metadata filters if available
            if debtor_country or creditor_country:
                countries = []
                if debtor_country:
                    countries.append(debtor_country)
                if creditor_country and creditor_country != debtor_country:
                    countries.append(creditor_country)
                    
                country_placeholders = ', '.join(['%s'] * len(countries))
                query += f" AND (e.country IS NULL OR e.country IN ({country_placeholders}))"
                params.extend(countries)
                
            if channel:
                query += " AND (e.payment_channel IS NULL OR e.payment_channel = %s)"
                params.append(channel)
                
            # Execute query
            records = self.db.execute_query(query, params)
            
            
            # Process results
            whitelist_matches = []
            blacklist_matches = []
            watchlist_matches = []
            
            for record in records:
                # Skip if missing key information
                if 'list_type' not in record or 'information_type' not in record or 'information_value' not in record:
                    continue
                    
                list_type = record['list_type']
                info_type = record['information_type']
                info_value = record['information_value'].lower() if record['information_value'] else ''
                
                # Find the matching identifier from our list
                matching_role = None
                for id_type, id_value, role in identifiers:
                    if id_type.lower() == info_type.lower() and id_value.lower() == info_value.lower():
                        matching_role = role
                        break
                        
                if not matching_role:
                    continue
                    
                # Create match entry
                match_entry = {
                    'entity_id': record.get('list_entity_id'),
                    'entity_name': record.get('entity_name', info_value),
                    'identifier': info_value,
                    'entity_type': record.get('entity_type'),
                    'category': record.get('category'),
                    'notes': record.get('notes'),
                    'risk_score': float(record.get('risk_score', 0)),
                    'risk_level': record.get('risk_level', 'LOW').upper(),
                    'match_type': f"{matching_role.upper()}_MATCH"
                }
                
                # Add to appropriate list
                if list_type == 'whitelist':
                    whitelist_matches.append(match_entry)
                elif list_type == 'blacklist':
                    blacklist_matches.append(match_entry)
                elif list_type == 'watchlist':
                    watchlist_matches.append(match_entry)
                    
            # Set results based on matches
            if whitelist_matches:
                results['whitelist']['is_whitelisted'] = True
                results['whitelist']['matched_entry'] = whitelist_matches[0]
                results['whitelist']['match_type'] = whitelist_matches[0]['match_type']
                
            if blacklist_matches:
                results['blacklist']['is_blacklisted'] = True
                results['blacklist']['matched_entry'] = blacklist_matches[0]
                results['blacklist']['match_type'] = blacklist_matches[0]['match_type']
                results['blacklist']['risk_score'] = blacklist_matches[0]['risk_score']
                results['blacklist']['risk_level'] = blacklist_matches[0]['risk_level']
                
                # Set severity based on risk level
                risk_level = blacklist_matches[0]['risk_level'].lower()
                if risk_level == 'critical':
                    results['blacklist']['severity'] = 'CRITICAL'
                elif risk_level == 'high':
                    results['blacklist']['severity'] = 'HIGH'
                elif risk_level == 'medium':
                    results['blacklist']['severity'] = 'MEDIUM'
                    
            if watchlist_matches:
                results['watchlist']['is_watchlisted'] = True
                results['watchlist']['matched_entry'] = watchlist_matches[0]
                results['watchlist']['match_type'] = watchlist_matches[0]['match_type']
                results['watchlist']['risk_score'] = watchlist_matches[0]['risk_score']
                results['watchlist']['risk_level'] = watchlist_matches[0]['risk_level']
                
            return results
            
        except Exception as e:
            self.logger.error(f"Error in combined platform list check: {str(e)}")
            return results

    def _process_database_operations(self, transaction: Dict, result: Dict, xml_content: str) -> None:
        """Process database operations with error handling"""

        risk_score = result.get('final_risk_score')
        risk_factors = result.get('risk_factors', {})
        # First collect all alerts
        all_alerts = []
    
        # Ensure fraud_types exists in the result
        if 'fraud_types' not in result:
            result['fraud_types'] = []

        if 'mule_account_indicators' in risk_factors:
            mule_risk_threshold = float(self.config_reader.get_property('mule.fraud.threshold', '60'))
            mule_account_risk = risk_factors['mule_account_risk']

            if risk_factors.get('mule_account_risk_score', 0) >= mule_risk_threshold:
                all_alerts.append({"type": "MULE", "message": "Possible Mule Account Activity"})

                # Add alerts for each risk factor
                for factor in mule_account_risk.get('factors', []):
                    all_alerts.append({"type": "MULE", "message": f"Indicator: {factor}"})

                # Also check for triggered_indicators which has more details
                if 'triggered_indicators' in risk_factors:
                    for indicator in risk_factors['triggered_indicators']:
                        all_alerts.append({
                            "type": "MULE", 
                            "message": f"Indicator: {indicator.get('details', '')}"
                        })
        
        # Alternative approach for legacy format
        if 'mule_account_indicators' in risk_factors:
            mule_risk_threshold = float(self.config_reader.get_property('mule.fraud.threshold', '60'))
            if risk_factors.get('mule_account_risk_score', 0) >= mule_risk_threshold:
                all_alerts.append({"type": "MULE", "message": "Possible Mule Account Activity"})
                
                for indicator in risk_factors['mule_account_indicators']:
                    all_alerts.append({"type": "MULE", "message": f"Indicator: {indicator.get('details', '')}"})
        
        if 'mule_relationship_indicators' in risk_factors:
            mule_relationship_threshold = float(self.config_reader.get_property('mule.relationship.threshold', '70'))
            if risk_factors.get('mule_relationship_risk_score', 0) >= mule_relationship_threshold:
                all_alerts.append({"type": "MULE_RELATIONSHIP", "message": "Possible Mule Account Relationship Detected"})
                
                for indicator_name, is_triggered in risk_factors['mule_relationship_indicators'].items():
                    if is_triggered:
                        all_alerts.append({"type": "MULE_RELATIONSHIP", "message": f"Indicator: {indicator_name}"})
        
            
        # Add account takeover fraud types if risk score is significant
        if 'account_takeover_indicators' in risk_factors:
            ato_risk_threshold = float(self.config_reader.get_property('ato.fraud.threshold', '60'))
            if risk_factors.get('account_takeover_risk_score', 0) >= ato_risk_threshold:

                if risk_factors.get('account_takeover_risk_score', 0) >= ato_risk_threshold:
                    all_alerts.append({"type": "ATO", "message": "Possible Account Takeover Activity"})
                
                for indicator in risk_factors['account_takeover_indicators']:
                    all_alerts.append({"type": "ATO", "message": f"Indicator: {indicator.get('details', '')}"})

        # Add APP fraud indicators if present
        if 'app_fraud_indicators' in risk_factors or 'app_behavioral_flags' in risk_factors:
            # First check the newer 'app_fraud_indicators' field
            if 'app_fraud_indicators' in risk_factors:
                for indicator in risk_factors['app_fraud_indicators']:
                    all_alerts.append({"type": "APP_FRAUD", "message": f"APP FRAUD: {indicator.get('details', '')}"})
                    # result['fraud_types'].append(f"APP FRAUD: {indicator.get('details', '')}")
            
            # Also check the app_behavioral_flags field
            elif 'app_behavioral_flags' in risk_factors:
                for flag in risk_factors['app_behavioral_flags']:
                    all_alerts.append({"type": "APP_FRAUD", "message": f"APP FRAUD: {flag.get('details', '')}"})
                    # result['fraud_types'].append(f"APP FRAUD: {flag.get('details', '')}")

            # Now process all alerts and add them to fraud_types
            for alert in all_alerts:
                result['fraud_types'].append(alert["message"])

        # For risk-based decisions, use the action from the decision controller
        action = result.get('action', 'PROCESS')

        if action in ['BLOCK_AND_REPORT', 'ALERT_AND_REVIEW']:
            # High or critical risk - generate high severity alert
            consolidated_alerts = []
            detailed_risk_message = f"RISK ASSESSMENT: (Transaction Risk Score: {risk_score:.1f}) | Contributing factors :"
                    
            # Initialize the consolidated_alerts list with the detailed risk message
            consolidated_alerts = [detailed_risk_message]

            # Group alerts by type and add them in order
            alert_types = ["MULE", "MULE_RELATIONSHIP", "ATO", "APP_FRAUD", "OTHER"]
           
            for alert_type in alert_types:
                type_alerts = [alert["message"] for alert in all_alerts if alert["type"] == alert_type]
                if type_alerts:
                    consolidated_alerts.extend(type_alerts)

            # Add any remaining alerts from result['alerts']
            if 'alerts' in result and isinstance(result['alerts'], list):
                for alert in result['alerts']:
                    if isinstance(alert, str):
                        consolidated_alerts.append(alert)
                    else:
                        # Handle the case when alert is not a string
                        consolidated_alerts.append(str(alert))
            
            # Add any fraud types that weren't already included
            if 'fraud_types' in result and isinstance(result['fraud_types'], list):
                for ft in result['fraud_types']:
                    if isinstance(ft, str) and ft not in consolidated_alerts:
                        consolidated_alerts.append(ft)
                
            # Add action information
            consolidated_alerts.append(f"Action: {action}")

            # Join all alerts with a separator
            alert_text = " | ".join(consolidated_alerts)

            # Insert fraud alert
            alert_id = self.db.insert_fraud_alert(
                transaction,
                risk_score,
                # result.get('final_risk_score', 0),
                alert_text,
                transaction.get('entity_id', 0),
                'Transaction Risk Alert',
                '',
                "HIGH"  # High severity for high risk
            )

            result['alert_generated'] = True
            
            # Increment alerts counter
            if hasattr(self, 'results'):
                if 'alerts_generated' in self.results:
                    self.results['alerts_generated'] += 1
            
            # Insert processed message
            self.db.insert_processed_message(
                transaction,
                alert_id,
                risk_score,
                # result.get('final_risk_score', 0),
                alert_text,
                xml_content
            )
        elif action == 'FLAG_FOR_REVIEW':
            # Medium risk case
            consolidated_alerts = []
            detailed_risk_message=f"RISK ASSESSMENT: (Transaction Risk Score: {risk_score:.1f}) | Contributing factors :"
                    
            # Initialize the consolidated_alerts list with the detailed risk message
            consolidated_alerts = [detailed_risk_message]

            # Group alerts by type and add them in order
            alert_types = ["MULE", "MULE_RELATIONSHIP", "ATO", "APP_FRAUD", "OTHER"]
           
            for alert_type in alert_types:
                type_alerts = [alert["message"] for alert in all_alerts if alert["type"] == alert_type]
                if type_alerts:
                    consolidated_alerts.extend(type_alerts)

            # Add any remaining alerts from result['alerts']
            if 'alerts' in result and isinstance(result['alerts'], list):
                for alert in result['alerts']:
                    if isinstance(alert, str):
                        consolidated_alerts.append(alert)
                    else:
                        # Handle the case when alert is not a string
                        consolidated_alerts.append(str(alert))
            
            # Add any fraud types that weren't already included
            if 'fraud_types' in result and isinstance(result['fraud_types'], list):
                for ft in result['fraud_types']:
                    if isinstance(ft, str) and ft not in consolidated_alerts:
                        consolidated_alerts.append(ft)
            
            # Add action information
            consolidated_alerts.append("Action: FLAG_FOR_REVIEW")
            
            # Join all alerts with a separator
            alert_text = " | ".join(consolidated_alerts) + " | REVIEW RECOMMENDED"
            
            # Insert a medium-severity alert
            alert_id = self.db.insert_fraud_alert(
                transaction,
                risk_score,
                # result.get('final_risk_score', 0),
                alert_text,
                transaction.get('entity_id', 0),
                'Transaction Risk Alert',
                '',
                "MEDIUM"  # Medium severity
            )
            result['alert_generated'] = True
            
            # Increment alerts counter
            if hasattr(self, 'results'):
                if 'alerts_generated' in self.results:
                    self.results['alerts_generated'] += 1

            # Insert processed message
            self.db.insert_processed_message(
                transaction,
                alert_id,
                risk_score,
                # result.get('final_risk_score', 0),
                alert_text,
                xml_content
            )
        else:
            # Low risk case (MONITOR, LOG, PROCESS)
            alert_id = None
            consolidated_alerts = []
            detailed_risk_message=f"RISK ASSESSMENT: (Transaction Risk Score: {risk_score:.1f}) | Contributing factors :"
                    
            # Initialize the consolidated_alerts list with the detailed risk message
            consolidated_alerts = [detailed_risk_message]
            
            # Group alerts by type and add them in order
            alert_types = ["MULE", "MULE_RELATIONSHIP", "ATO", "APP_FRAUD", "OTHER"]
           
            for alert_type in alert_types:
                type_alerts = [alert["message"] for alert in all_alerts if alert["type"] == alert_type]
                if type_alerts:
                    consolidated_alerts.extend(type_alerts)

            # Add any remaining alerts from result['alerts']
            if 'alerts' in result and isinstance(result['alerts'], list):
                for alert in result['alerts']:
                    if isinstance(alert, str):
                        consolidated_alerts.append(alert)
                    else:
                        # Handle the case when alert is not a string
                        consolidated_alerts.append(str(alert))
            
            # Add any fraud types that weren't already included
            if 'fraud_types' in result and isinstance(result['fraud_types'], list):
                for ft in result['fraud_types']:
                    if isinstance(ft, str) and ft not in consolidated_alerts:
                        consolidated_alerts.append(ft)

            
            # Add action information
            if action in ['MONITOR', 'LOG']:
                consolidated_alerts.append(f"Action: {action}")

            # Join all alerts with a separator
            alert_text = " | ".join(consolidated_alerts) if consolidated_alerts else f"No elevated risk detected | Action: {action}"

            # Insert processed message
            self.db.insert_processed_message(
                transaction,
                alert_id,
                risk_score,
                # result.get('final_risk_score', 0),
                alert_text,
                xml_content
            )

        # Now log the transaction
        # try:
        #     self.transaction_logger.log_transaction(transaction, result)
        # except Exception as e:
        #     self.logger.error(f"Error logging transaction: {str(e)}")

        self.generate_risk_assessment_json(transaction, result)



    def generate_risk_assessment_json(self,transaction, result):
        """
        Generate a JSON representation of risk assessment results by simply including the entire
        transaction and result objects as they exist.    
        """
        # Create JSON output with the complete objects
        json_output = {
            "transaction": transaction,
            "alert_type" : 'Transaction Risk Alert',
            "result": result
        }
 
        return json_output

    def process_message(self, xml_content: str, channel: str, entity_id: int, file_path: Path) -> Dict:
        """
        Process ISO20022 message with consolidated fraud detection checks.
        """
        try:
            result = self._initialize_result()
            
            # Parse message
            try:
                parsed_message = self.parser.parse_message(xml_content)
                if not parsed_message:
                    raise ValueError("Failed to parse message")
                    
                transaction = self.parser._convert_to_transaction_format(parsed_message)
                if not transaction:
                    raise ValueError("Failed to convert transaction format")
                
            except Exception as e:
                logger.error(f"Error parsing message: {str(e)}")
                return result
             # Set transaction ID and basic info
            result['transaction_id'] = transaction.get('msg_id', 'UNKNOWN')
            transaction['channel'] = channel
            transaction['entity_id'] = entity_id
            
            # Ensure creation_date is set
            if 'creation_date' not in transaction:
                transaction['creation_date'] = datetime.now()

            # Process compliance checks
            compliance_result = self._process_compliance_checks(transaction, result)

            if compliance_result is None:
                logger.error("Compliance checks returned None")
                return result
            elif compliance_result.get('early_return'):
                return compliance_result.get('result', result)

            # Process fraud detection
            start_time= time.time()
            fraud_result = self._process_fraud_detection(transaction, result)
            if fraud_result.get('error'):
                return fraud_result.get('result', result)
            end_time=time.time()
            elapsed = end_time - start_time
            # print(f"Elapsed time in process_fraud_detection-XML: {elapsed:.6f} seconds")
            # NEW: Apply decision controller to get risk-based actions
            result = self.decision_controller.process_transaction_result(result)

            # Process database operations
            try:
                self._process_database_operations(transaction, result, xml_content)
                # result['alert_generated'] = True
            except Exception as e:
                # result['alert_generated'] = False
                self.logger.error(f"Error storing XML transactions in DB : {str(e)}")

            # Add additional metadata for pattern discovery
            result['raw_xml'] = xml_content
            result['channel'] = channel
            result['entity_id'] = entity_id
            result['timestamp'] = datetime.now()

            return result

        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return self._initialize_result()

        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return {
                'transaction_id': 'UNKNOWN',
                # 'is_suspicious': False,
                'final_risk_score': 0,
                'alerts': [],
                'metrics': {},
                'risk_factors': {
                    "compliance_alerts": [],
                    "velocity_alerts": [],
                    "amount_alerts": [],
                    "geographic_alerts": [],
                    "recipient_alerts": [],
                    "pattern_alerts": [],
                    "channel_alerts": [],
                    "triggered_rules": []
                }
            }

    def _prepare_features_dict(self, transaction: Dict, metrics: TransactionMetrics) -> Dict:
        """Prepare features dictionary with safe handling of None values"""
        try:
            # Get country code and transaction time (safe extraction)
            country_code = transaction.get('debtor_country', 'UNKNOWN')
            transaction_time = transaction.get('creation_date', datetime.now())
            
            # Use the is_weekend field that we added in row_to_transaction
            is_weekend = transaction.get('is_weekend', False)
            local_hour = transaction.get('local_hour', datetime.now().hour)
            is_business_hours = transaction.get('is_business_hours', False)
            is_high_risk_hour = transaction.get('is_high_risk_hour', True)

            # Build the features dictionary with safe defaults for all fields
            features_dict = {
                'amount': float(transaction.get('amount', 0)),
                'amount_zscore': 0.0,  # Default value
                'tx_count_1h': metrics.velocity_24h / 24 if hasattr(metrics, 'velocity_24h') else 0,
                'tx_count_24h': metrics.velocity_24h if hasattr(metrics, 'velocity_24h') else 0,
                'route_count_1h': metrics.unique_recipients_24h / 24 if hasattr(metrics, 'unique_recipients_24h') else 0,
                'route_count_24h': metrics.unique_recipients_24h if hasattr(metrics, 'unique_recipients_24h') else 0,
                'cross_border': metrics.cross_border if hasattr(metrics, 'cross_border') else False,
                'high_risk_route': metrics.high_risk_country if hasattr(metrics, 'high_risk_country') else False,
                'is_high_risk_hour': is_high_risk_hour,
                'is_business_hours': is_business_hours,
                'hour': local_hour,
                'is_weekend': is_weekend,
                'debtor_country': country_code,
                'creditor_country': transaction.get('creditor_country', 'UNKNOWN')
            }
            # Only calculate amount_zscore if we have valid metrics
            if hasattr(metrics, 'avg_amount_30d') and hasattr(metrics, 'std_amount_30d'):
                # Safe calculation of z-score
                if metrics.std_amount_30d > 0:
                    features_dict['amount_zscore'] = (float(transaction.get('amount', 0)) - metrics.avg_amount_30d) / metrics.std_amount_30d
                else:
                    features_dict['amount_zscore'] = 0.0

            # Update velocity metrics if available
            if hasattr(metrics, 'velocity_24h'):
                features_dict['tx_count_1h'] = metrics.velocity_24h / 24
                features_dict['tx_count_24h'] = metrics.velocity_24h
                
            # Update recipient metrics if available
            if hasattr(metrics, 'unique_recipients_24h'):
                features_dict['route_count_1h'] = metrics.unique_recipients_24h / 24
                features_dict['route_count_24h'] = metrics.unique_recipients_24h
                
            # Update cross-border and high-risk route fields if available
            if hasattr(metrics, 'cross_border'):
                features_dict['cross_border'] = metrics.cross_border
                
            if hasattr(metrics, 'high_risk_country'):
                features_dict['high_risk_route'] = metrics.high_risk_country
            
            return features_dict
    
        
        except Exception as e:
            logger.error(f"Error preparing features dictionary: {str(e)}")
            # Return a minimal valid dictionary to prevent further errors
            return {
                'amount': float(transaction.get('amount', 0)),
                'tx_count_24h': 0,
                'is_high_risk_hour': True,
                'is_business_hours': False,
                'hour': datetime.now().hour,
                'is_weekend': False,
                'debtor_country': 'UNKNOWN',
                'creditor_country': 'UNKNOWN',
                'amount_zscore': 0.0,
                'tx_count_1h': 0,
                'route_count_1h': 0,
                'route_count_24h': 0,
                'cross_border': False,
                'high_risk_route': False
            }
        
    def _improved_update_result_with_computed_values(self, result: Dict, transaction: Dict,
                                  metrics: TransactionMetrics, ml_score: float,
                                  channel_score: float, app_fraud_results: Dict,
                                  risk_analysis: Dict, calc_final_risk_score: float) -> None:
        """Enhanced version with better score recording for analysis"""
        try:
            # Use original implementation
            self._update_result_with_computed_values(
                result, transaction, metrics, ml_score,
                channel_score, app_fraud_results, risk_analysis, calc_final_risk_score
            )
            
            # Add additional details for analysis
            # Store all score components in model_scores for easier analysis
            result['model_scores'] = {
                'ml_score': ml_score,
                'consolidated_score': risk_analysis.get('consolidated_score', 0),
                'channel_score': channel_score,
                'app_fraud_score': app_fraud_results.get('risk_score', 0),
                'mule_account_score': risk_analysis.get('mule_indicators_triggered', 0) * 10,  # Approximate
                'ato_score': risk_analysis.get('ato_indicators_triggered', 0) * 10,  # Approximate
                # Add all risk components for detailed analysis
                'risk_components': {
                    comp: details.get('score', 0) 
                    for comp, details in risk_analysis.get('risk_components', {}).items()
                }
            }
            
            # Add detailed feature information for potential analysis
            try:
                features_dict = self._prepare_features_dict(transaction, metrics)
                # Keep only numeric features for simplicity
                result['feature_values'] = {
                    k: v for k, v in features_dict.items() 
                    if isinstance(v, (int, float, bool))
                }
            except Exception as e:
                logger.debug(f"Could not extract feature values: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error in enhanced update of result values: {str(e)}")
            
    def _update_result_with_computed_values(self, result: Dict, transaction: Dict,
                                      metrics: TransactionMetrics, ml_score: float,
                                      channel_score: float, app_fraud_results: Dict,
                                      risk_analysis: Dict, calc_final_risk_score: float) -> None:
        """Update result dictionary with computed values"""
        try:
            # STEP 1: Apply ML model confidence boost - Not being done
            # ml_score_scaled = ml_score  # Keep original scale for component tracking
            # ml_score_boosted = ml_score
            
        
            # STEP 2: Apply velocity and pattern importance weights
            # consolidated_score = risk_analysis.get('consolidated_score', 0)
            consolidated_score = calc_final_risk_score
            velocity_importance = float(self.config_reader.get_property('risk.velocity_importance', '1.2'))
            pattern_importance = float(self.config_reader.get_property('risk.pattern_importance', '1.15'))

            # Extract components from risk analysis
            velocity_component = 0
            pattern_component = 0
            for component_name, details in risk_analysis.get('risk_components', {}).items():
                if 'velocity' in component_name.lower():
                    velocity_component = details.get('score', 0) * velocity_importance
                if 'pattern' in component_name.lower() or 'account_takeover' in component_name.lower():
                    pattern_component = details.get('score', 0) * pattern_importance

            # Enhance score if velocity or pattern signals are strong
            enhanced_score = consolidated_score
            if velocity_component > 50 or pattern_component > 50:
                enhanced_score = min(enhanced_score * 1.1, 100)
                logger.info(f"Enhancing score due to strong velocity ({velocity_component}) or pattern ({pattern_component}) signals")
            
            # STEP 3: Calculate final risk score, combining all components
            final_risk_score = enhanced_score 
            
            # STEP 4: Update risk factors
            risk_factors = result.get('risk_factors', {})
            risk_factors.update({
                "risk_components": risk_analysis.get('risk_components', {}),
                "contributing_factors": risk_analysis.get('contributing_factors', []),
                "ato_indicators": risk_analysis.get('ato_indicators_triggered', 0),
                "mule_indicators": risk_analysis.get('mule_indicators_triggered', 0),
                "app_fraud_patterns": app_fraud_results.get('identified_patterns', []),
                "app_behavioral_flags": app_fraud_results.get('behavioral_flags', []),
                "app_fraud_score": app_fraud_results.get('risk_score', 0),
                "score_components": {
                    "base_score": risk_analysis.get('base_score', 0),
                    "high_risk_penalty": risk_analysis.get('high_risk_penalty', 0),
                    "ml_score": ml_score,  # Convert to 0-100 scale for display
                    "consolidated_score": consolidated_score,
                    "enhanced_score": enhanced_score,
                    "velocity_component": velocity_component,
                    "pattern_component": pattern_component,
                    "channel_score": channel_score,
                    "app_fraud_score": app_fraud_results.get('risk_score', 0),
                    "final_score": final_risk_score
                }
            })

            # STEP 5: Update fraud types
            fraud_types = risk_analysis.get('fraud_types', [])
            
            # Add APP fraud types if risk score is significant
            app_fraud_risk_score = app_fraud_results.get('risk_score', 0)
            if app_fraud_risk_score >= float(self.config_reader.get_property('app.fraud.threshold', '70')):
                # Add fraud types for identified patterns
                for pattern in app_fraud_results.get('identified_patterns', []):
                    if pattern.get('type'):
                        fraud_type = f"APP_FRAUD_{pattern.get('type').upper()}"
                        if fraud_type not in fraud_types:
                            fraud_types.append(fraud_type)
                
                # Add fraud types for behavioral flags
                for flag in app_fraud_results.get('behavioral_flags', []):
                    if flag.get('type'):
                        fraud_type = f"APP_FRAUD_BEHAVIOR_{flag.get('type').upper()}"
                        if fraud_type not in fraud_types:
                            fraud_types.append(fraud_type)
            
        
            # Add mule account fraud types if risk score is significant
            if 'mule_account_indicators' in risk_factors:
                mule_risk_threshold = float(self.config_reader.get_property('mule.fraud.threshold', '60'))
                if result.get('risk_factors', {}).get('mule_account_risk_score', 0) >= mule_risk_threshold:
                    fraud_types.append("MULE_ACCOUNT_ACTIVITY")

            # Format mule relationship details if present
            if 'mule_relationship_details' in result['risk_factors']:
                # Extract key insights from mule relationship details
                mule_details = result['risk_factors']['mule_relationship_details']

                # Add formatted information to the result
                result['summary'].append(f"Detected {len(mule_details.get('patterns_detected', []))} mule network patterns")
                
                # Add relationship graph information if available
                if 'relationship_graph' in mule_details:
                    graph_info = mule_details['relationship_graph']
                    result['relationship_graph'] = {
                        'nodes': len(graph_info.get('nodes', [])),
                        'connections': len(graph_info.get('edges', [])),
                        'high_risk_entities': graph_info.get('high_risk_count', 0)
                    }
                
                # Add detection time and confidence level
                result['mule_detection_metadata'] = {
                    'detection_time': datetime.now().isoformat(),
                    'confidence_level': result['risk_factors'].get('mule_relationship_score', 0)
                }

                    
            # Add account takeover fraud types if risk score is significant
            if 'account_takeover_indicators' in risk_factors:
                ato_risk_threshold = float(self.config_reader.get_property('ato.fraud.threshold', '60'))
                if result.get('risk_factors', {}).get('account_takeover_risk_score', 0) >= ato_risk_threshold:
                    fraud_types.append("ACCOUNT_TAKEOVER_ACTIVITY")
            
            # STEP 7: Use DecisionController for final determination
            # This should be the single source of truth for risk level and action
            result['final_risk_score'] = final_risk_score
            result['risk_level'] = self.decision_controller.get_risk_level(final_risk_score)
            result['action'] = self.decision_controller.get_action(final_risk_score)
            
            
            # STEP 7.1 : Create a comprehensive risk summary for database storage
            detailed_risk_message = ""
            detailed_risk_message=f"RISK ASSESSMENT: {result['final_risk_score']} (Score: {consolidated_score:.1f}) | "
            # Add fraud types if any
            if fraud_types:
                detailed_risk_message += f"Fraud Types: {', '.join(fraud_types)} | "
            
            if result.get('contributing_factors'):
                top_factors = result.get('contributing_factors')[:5]  # Limit to 5 most important factors
                detailed_risk_message += f"Key Factors: {'; '.join(top_factors)} | "

            # STEP 8: Update the complete result
            result['fraud_types'] = fraud_types
            result['risk_factors'] = risk_factors
            result['channel'] = transaction.get('channel', 'UNKNOWN')
            # result['metrics'] = asdict(metrics) if metrics else {}
            if metrics:
                if is_dataclass(metrics):
                    result['metrics'] = asdict(metrics)
                else:
                    # Fallback: convert to dictionary if it has a __dict__ attribute
                    result['metrics'] = metrics.__dict__ if hasattr(metrics, '__dict__') else {}
            else:
                result['metrics'] = {}
            result['timestamp'] = datetime.now().isoformat()
        

        except Exception as e:
            logger.error(f"Error updating result with computed values: {str(e)}")
    

def load_trained_models(system, timestamp, model_dir="./trained_models"):
    """
    Load trained models into an existing FraudDetectionSystem instance
    
    :param system: FraudDetectionSystem instance
    :param timestamp: Timestamp string used when models were saved
    :param model_dir: Directory containing the saved models
    :return: Tuple of (success_flag, error_message)
    """
    try:
        import pickle
        import os
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info(f"Loading models with timestamp {timestamp} from {model_dir}")
        
        # Load feature extractor
        extractor_path = f"{model_dir}/feature_extractor_{timestamp}.pkl"
        if not os.path.exists(extractor_path):
            return False, f"Feature extractor file not found: {extractor_path}"
            
        with open(extractor_path, 'rb') as f:
            system.feature_extractor = pickle.load(f)
        logger.info("Feature extractor loaded successfully")
        
        # Load models
        model_names = ['random_forest', 'neural_net', 'xgboost']
        loaded_models = {}
        
        for name in model_names:
            model_path = f"{model_dir}/{name}_model_{timestamp}.pkl"
            if not os.path.exists(model_path):
                return False, f"Model file not found: {model_path}"
                
            with open(model_path, 'rb') as f:
                loaded_models[name] = pickle.load(f)
            logger.info(f"Model {name} loaded successfully")

            system.feature_extractor.config_reader = system.config_reader
            system.feature_extractor.channel_detector = system.channel_detector
            # print(f"DEBUG: Re-set feature_extractor.config_reader and channel_detector after loading from pickle")
        # Update instance models
        system.models = loaded_models
        
        logger.info("All models loaded successfully")
        return True, "Models loaded successfully"
        
    except Exception as e:
        error_message = f"Error loading models: {str(e)}"
        logger.error(error_message)
        return False, error_message

def send_license_notification(days_remaining, expiry_date, config_reader):

    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    

    """Send email notification about license expiration"""
    # Email configuration
    smtp_server = config_reader.get_property('smtp.server', "smtp.gmail.com")
    smtp_port = config_reader.get_property('smtp.port', "587")
    smtp_user = config_reader.get_property('smtp.user', "bliinksmtpuser@gmail.com")  # Replace with your email
    smtp_user_password = config_reader.get_property('smtp.password', "tbty pojc pulh iyyk")   # Replace with your app password
    sender_email = config_reader.get_property('smtp.sender_email', "reetesh.ghosh@blinkaipayments.com")
    recipient_email = config_reader.get_property('smtp.receiver_email', "kishore.joseph@blinkaipayments.com")  # Replace with recipient email

    # Read app information from config
    app_name = config_reader.get_property('app.name', 'BliinkAI Model')
    app_version = config_reader.get_property('app.version', '1.0.0')

    # Create message
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = recipient_email
    message["Subject"] = f"{app_name}-ver{app_version} License Expiring in {days_remaining} days"

    # Email body
    body = f"""

    IMPORTANT: BliinkAI License Expiration Notice

    Your BliinkAI software license is expiring soon.

    Days Remaining: {days_remaining}
    Expiry Date: {expiry_date}

    Please contact the BliinkAI Sales team to renew your license and ensure uninterrupted service.

    For assistance, please contact:
    Email: sales@blinkaipayments.com
    Phone: +1-XXX-XXX-XXXX

    Best regards,
    BliinkAI Team
    """

    message.attach(MIMEText(body, "plain"))

    print(f"message is {message}")
    try:
        # Create SMTP session
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.set_debuglevel(1) 
        server.starttls()
        server.login(smtp_user, smtp_user_password)
        
        # Send email
        # server.sendmail(sender_email,recipient_email,message)
        server.send_message(message)
        server.quit()
        print(f"License expiration notification sent to {recipient_email}")
        
    except Exception as e:
        print(f"Failed to send license notification email: {str(e)}")

def check_license(config_reader):
    """Check if license is valid and not expired"""
    from cryptography.fernet import Fernet
    from datetime import datetime
    import os
    import sys
    import json

    LICENSE_FILE = 'license.dat'
    ENCRYPTED_KEY = b'XHswOl8sUy0pPW4pe3e3i6KVP6MlLa3fyGJHZWjiqGk='

    try:
        # Check if license file exists
        if not os.path.exists(LICENSE_FILE):
            print("License file not found! Please contact BliinkAI Sales team.")
            sys.exit(1)

        # Use embedded key instead of reading from file
       
        fernet = Fernet(ENCRYPTED_KEY)

        # Read and decrypt license data
        with open(LICENSE_FILE, 'rb') as license_file:
            encrypted_data = license_file.read()
            decrypted_data = fernet.decrypt(encrypted_data)
            license_data = json.loads(decrypted_data.decode())

        # Get expiry date from license
        expiry_date = datetime.strptime(license_data['expiry_date'], '%Y-%m-%d').date()
        today = datetime.now().date()

        if today > expiry_date:
            print("\nSTATUS: LICENSE EXPIRED")
            print(f"License expired {(today - expiry_date).days} days ago")
            print("Please contact BliinkAI Sales team for renewal.")
            sys.exit(1)

        # License is valid
        days_remaining = (expiry_date - today).days
        print("\nSTATUS: LICENSE VALID")
        print(f"Days remaining: {days_remaining}")
        # Send notification if 30 or fewer days remaining
        if days_remaining <= 30:
            send_license_notification(days_remaining, expiry_date,config_reader)

        return days_remaining

    except Exception as e:
        print("Error validating license! Please contact BliinkAI Sales team.")
        print(f"Error details: {str(e)}")
        sys.exit(1)

def setup_logging_from_config(config_file):
    """Sets up logging from a configuration file using ConfigReader."""

    try:
        config = ConfigReader(config_file)
    except FileNotFoundError:
        print(f"Error: Config file '{config_file}' not found.")
        return  # Or handle the error as needed
    except Exception as e: # Catch other potential configreader errors
        print(f"Error reading config file: {e}")
        return

    # Create logs directory if it doesn't exist
    log_dir = config.get_property('log.directory', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Get log level from config, default to INFO if not specified
    log_level_str = config.get_property('log.level', 'INFO').upper()
    
    # Map log level string to logging constant
    log_level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    # Get the log level, default to INFO if not found
    log_level = log_level_map.get(log_level_str, logging.INFO)
    
    # Create a unique log filename with timestamp
    log_filename = os.path.join(log_dir, f'fraud_detection_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    # Create handlers
    file_handler = RotatingFileHandler(
        log_filename, 
        maxBytes=int(config.get_property('log.max_file_size', 10*1024*1024)),  # 10 MB default
        backupCount=int(config.get_property('log.backup_count', 5))  # 5 backup files default
    )
    console_handler = logging.StreamHandler()
    
    # Create formatters
    formatter = logging.Formatter(
        config.get_property(
            'log.format', 
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Set levels for handlers
    file_handler.setLevel(log_level)
    console_handler.setLevel(log_level)
    
    # Configure the root logger
    logging.basicConfig(
        level=log_level,
        handlers=[file_handler, console_handler]
    )
    
    # Log the configuration
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured:")
    logger.info(f"Log Level: {log_level_str}")
    logger.info(f"Log File: {log_filename}")
    
    return log_filename

def process_transaction_from_kafka(xml_content: str) -> dict:
    """
    Process XML content received directly from Kafka.
    """
    logger = logging.getLogger(__name__)
    # Start timing
    start_time = time.time()
    
    # Component timing
    timings = {
        'setup': 0,
        'parsing': 0,
        'file_op': 0,
        'processing': 0,
        'total': 0
    }
    
    # Initialize results
    results = {
        'processed': False,
        'is_suspicious': False,
        'alerts_generated': 0,
        'error': None
    }
    
    # Temporary file path for compatibility with process_message
    temp_path = None
    
    try:
        # Get singleton instances - avoid creating new objects for each transaction
        t1 = time.time()
        system, config_reader = get_fraud_system()
        
        # Quick license check (should be cached by the singleton)
        if not check_license(config_reader):
            logger.error("License check failed.")
            results['error'] = "License check failed"
            return results
        
        timings['setup'] = time.time() - t1
        
        # Parse XML with minimal normalization (avoid pretty printing)
        t1 = time.time()
        # Only add namespace if needed and without pretty printing
        if "<Document" in xml_content and 'xmlns:pain' not in xml_content:
            # Simple string replacement is much faster than DOM parsing
            if 'xmlns=' not in xml_content:
                xml_content = xml_content.replace(
                    "<Document", 
                    '<Document xmlns="urn:iso:std:iso:20022:tech:xsd:pain.001.001.09"'
                )
        
        # Handle entity_id extraction with fallback
        try:
            entity_id = system._extract_entity_id(xml_content)
        except Exception as e:
            logger.warning(f"Error extracting entity_id: {e}. Using default value.")
            entity_id = 0
        
        # Parse the message
        try:
            parsed_message = system.parser.parse_message(xml_content)
            channel = parsed_message['payment_channel'].value
        except ValueError as ve:
            if "Group Header not found" in str(ve):
                logger.error("XML parsing failed: Group Header not found.")
                results['error'] = "Group Header not found in XML. Format mismatch."
                return results
            else:
                logger.error(f"XML parsing failed: {ve}")
                results['error'] = str(ve)
                return results
        except Exception as e:
            logger.error(f"Error parsing message: {e}")
            results['error'] = str(e)
            return results
        timings['parsing'] = time.time() - t1
        
        # Create temporary file (required by process_message)
        t1 = time.time()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xml') as temp:
            temp.write(xml_content.encode('utf-8'))
            temp_path = temp.name
        file_path = Path(temp_path)
        timings['file_op'] = time.time() - t1
        
        # Process the message
        t1 = time.time()
        try:
            result = system.process_message(xml_content, channel, entity_id, file_path)
            
            # Update results
            results.update({
                'processed': True,
                'is_suspicious': result.get('is_suspicious', False),
                'alerts_generated': 1 if result.get('is_suspicious', False) else 0,
                'transaction_id': result.get('transaction_id'),
                'entity_id': entity_id
            })
        except Exception as e:
            logger.error(f"Error in process_message: {e}")
            results['error'] = str(e)
        timings['processing'] = time.time() - t1
        
        # Calculate total time
        # timings['total'] = time.time() - start_time
        
        # Add timing to results
        results['processing_time'] = timings['total']
        
        # Log timing breakdown to identify bottlenecks
        # Log timing breakdown to identify bottlenecks
        logger.info(f"Transaction {results.get('transaction_id', 'unknown')} processed in {int(timings['total']*1000)} ms")
        logger.info(f"Timing breakdown: setup={int(timings['setup']*1000)} ms, parsing={int(timings['parsing']*1000)} ms, " +
                    f"file_op={int(timings['file_op']*1000)} ms, processing={int(timings['processing']*1000)} ms")
        
        return results
    
    except Exception as e:
        # Log timing even for errors
        timings['total'] = time.time() - start_time
        logger.error(f"Error processing message: {str(e)}, time spent: {timings['total']:.3f}s", exc_info=True)
        results['error'] = str(e)
        return results
    
    finally:
        # Clean up the temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                logger.warning(f"Failed to remove temporary file: {e}")

_fraud_system = None
_config_reader = None

def get_fraud_system(train_data_path=None):
    """
    Get the fraud detection system instance and config reader.
    Uses singleton pattern to avoid re-initialization.
    Returns a tuple of (FraudDetectionSystem, ConfigReader)
    """
    global _fraud_system, _config_reader
    
    if _fraud_system is None or _config_reader is None:
        config_path = "./config.properties"
        _config_reader = ConfigReader(config_path)
        _config_reader.load_properties()
        
        # Check license first
        if not check_license(_config_reader):
            raise Exception("License check failed")
            
        # Initialize fraud detection system
        _fraud_system = FraudDetectionSystem(config_path)
        logger.info("Fraud detection system initialized")
        # Train models if data path is provided
        if train_data_path is not None and _fraud_system is not None:
            # Get save path from config or use default
            save_path = _config_reader.get_property('models.directory', './trained_models')
            
            try:
                logger.info(f"Training models with data from {train_data_path}...")
                
                
                # Train models and get evaluation results
                evaluation_results = _fraud_system.train_models_from_csv(
                    csv_path=train_data_path,
                    save_path=save_path
                )
                
                # Log evaluation results
                logger.info("Model training completed. Results:")
                for model_name, metrics in evaluation_results.items():
                    logger.info(f"  {model_name}:")
                    for metric_name, value in metrics.items():
                        logger.info(f"    {metric_name}: {value:.4f}")
                
                # Get current timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Update config with timestamp of newly trained models
                _config_reader.set_property('models.timestamp', timestamp)
                _config_reader.set_property('models.use_trained', 'true')
                _config_reader.save_properties()
                
                logger.info(f"Config updated with new model timestamp: {timestamp}")
                
                return _fraud_system, _config_reader, evaluation_results
            
            except Exception as e:
                logger.error(f"Error training models: {str(e)}")
                logger.info("Continuing with existing models")
        
    
    return _fraud_system, _config_reader



def main():
    """Main function for training and testing the system"""
    # Check for command line arguments
    import sys
    import subprocess
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()  # Convert to lowercase for case-insensitive comparison
        
        if command == "--admin":
            try:
                # Launch config editor
                subprocess.run(["python", "config_editor.py"], check=True)
                print("Config editor launched.")
                return  # Exit the main function after config editor completes
            except FileNotFoundError:
                print("Error: config_editor.py not found. Make sure it's in the correct directory.")
                sys.exit(1)
            except subprocess.CalledProcessError as e:
                print(f"Error running config_editor.py: {e}")
                sys.exit(1)
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                sys.exit(1)

        elif command == "--analyze":
                # Use the remaining arguments for analysis
                from analyze_predictions import main as analyze_main
                
                # Remove the --analyze flag to prepare args for analyze_main
                sys.argv.pop(1)  
                
                # Now call the analyze_main function with the modified argv
                analyze_main()
                return

        # If we get here, the command wasn't recognized
        print(f"Unknown command: {command}")
        print("Available commands: --admin, --analyze")
        sys.exit(1)

    # Continue with normal program execution if no admin flag or after admin tasks
    # Load config first
    import os
    print(f"Current working directory: {os.getcwd()}")
    config_path = "./config.properties"
    print(f"Config file exists: {os.path.exists(config_path)}")

    if os.path.exists(config_path):
    # Manually read the file to check contents
        with open(config_path, 'r') as f:
            config_contents = f.read()
        # print(f"Config file contents:\n{config_contents}")

    config_reader = ConfigReader("./config.properties")
    config_reader.load_properties()  
    # Setup logging using config reader
    log_file = setup_logging_from_config(config_reader.get_property('log.file', './fraud_detection.log'))
    
    # Get logger for this module
    logger = logging.getLogger(__name__)

    logger.info("Initializing Fraud Detection System...")

    # Check license first
    if not check_license(config_reader):
        sys.exit(1)
        

    try:
        
        # Get database type from config
        db_type = config_reader.get_property('database.type', 'sql')
        logger.info(f"Database type from config in main(): '{db_type}'")
        
        # Log database connection parameters (but mask password)
        db_host = config_reader.get_property('db.host', '')
        db_name = config_reader.get_property('db.database', '')
        db_user = config_reader.get_property('db.user', '')
        db_port = config_reader.get_property('db.port', '3306')
        
        logger.info(f"Attempting database connection with: host={db_host}, db={db_name}, user={db_user}, port={db_port}")
        
        # Check if the MySQL connector is available
        try:
            import mysql.connector
            logger.info("MySQL connector imported successfully")
        except ImportError:
            logger.error("MySQL connector not found. Please install with: pip install mysql-connector-python")
            sys.exit(1)
        
        # # Initialize FraudDetectionSystem with better error handling
        # Loading twice commented out
        # fraud_detection_system = FraudDetectionSystem(config_path)
    except ImportError as e:
        logger.error(f"Missing required module: {str(e)}")
        sys.exit(1)
    except mysql.connector.Error as e:
        logger.error(f"MySQL Error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to initialize Fraud Detection System: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

    # Check if we should load pre-trained models
        # use_trained_models = config_reader.get_property('models.use_trained', 'false') == 'true'
        # model_timestamp = config_reader.get_property('models.timestamp', '')
        # model_dir = config_reader.get_property('models.directory', './trained_models')

    # Initialize results tracking
    results = {
       'total_processed': 0,
        'failed_processing': 0,
        'alerts_generated': 0,
        'csv_processed': 0,
        'xml_processed': 0,
        'ach_processed': 0,
        'return_processed': 0,
        'return_matched': 0,
        'return_unmatched': 0
    }
    
    try:
        
        # Get database type from configuration
        db_type = config_reader.get_property('database.type','sql')
        logger.info(f"Database type from config in main(): '{db_type}'")
        # Test appropriate database connection based on configuration
        if db_type == 'csv':
            db = CSVDatabaseConnection()
        
        if db_type == 'sql':  
            db = DatabaseConnection()

        logger.info(f"Database connection ({db_type}) successful")
        db.close()


        # Check if training data path is specified
        training_data_path = "enhanced_dataset.csv"

        if training_data_path and os.path.exists(training_data_path):
            # Check if data file has changed since last training
            last_trained_timestamp = config_reader.get_property('models.last_data_timestamp', '0')
            data_file_timestamp = str(int(os.path.getmtime(training_data_path)))
            
            # Get information about the trained model
            use_trained_models = config_reader.get_property('models.use_trained', 'false') == 'true'
            model_timestamp = config_reader.get_property('models.timestamp', '')
            model_dir = config_reader.get_property('models.directory', './trained_models')
            
            # Determine if we need to retrain
            needs_training = False
            
            if not use_trained_models or not model_timestamp:
                # No trained model exists
                logger.info("No trained model exists, training required")
                needs_training = True
            elif data_file_timestamp != last_trained_timestamp:
                # Training data has changed since last training
                logger.info(f"Training data has changed (file: {data_file_timestamp}, last trained: {last_trained_timestamp})")
                needs_training = True
            else:
                logger.info("Training data has not changed since last model training")
            
            if needs_training:
                # Train the model
                logger.info(f"Training model with data from {training_data_path}")
                
                # Temporarily set use_trained to false to prevent loading existing models
                original_use_trained = config_reader.get_property('models.use_trained', 'false')
                config_reader.set_property('models.use_trained', 'false')
                config_reader.save_properties()
                
                try:
                    # Get system without loading models (since we just disabled model loading)
                    result = get_fraud_system(train_data_path=training_data_path)
                    
                    # Handle different return value counts from get_fraud_system
                    if isinstance(result, tuple) and len(result) == 3:
                        system, config_reader, evaluation_results = result
                    elif isinstance(result, tuple) and len(result) == 2:
                        system, config_reader = result
                        evaluation_results = {}  # Default empty evaluation results
                        logger.warning("No evaluation results returned from get_fraud_system")
                    else:
                        raise ValueError(f"Unexpected return format from get_fraud_system: {result}")
                    
                    # Get current timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    # Update config with both timestamps
                    config_reader.set_property('models.timestamp', timestamp)
                    config_reader.set_property('models.last_data_timestamp', data_file_timestamp)
                    config_reader.set_property('models.use_trained', 'true')
                    config_reader.save_properties()
                    
                    logger.info(f"Model trained and saved with timestamp {timestamp}")
                    logger.info(f"Training data timestamp recorded as {data_file_timestamp}")
                    
                    logger.info("Model evaluation results:")
                    for model_name, metrics in evaluation_results.items():
                        logger.info(f"  {model_name}: {metrics}")
                except Exception as e:
                    # If training fails, restore original setting
                    logger.error(f"Training failed: {str(e)}")
                    config_reader.set_property('models.use_trained', original_use_trained)
                    config_reader.save_properties()
                    raise
            else:
                # Just get the system without training
                logger.info(f"Using existing trained model with timestamp {model_timestamp}")
                system, config_reader = get_fraud_system()
                
                # Load the trained model
                success, message = load_trained_models(system, model_timestamp, model_dir)
                if success:
                    logger.info("Successfully loaded trained models")
                else:
                    logger.warning(f"Failed to load trained models: {message}. Will retrain.")
                    
                    # Temporarily set use_trained to false to prevent loading 
                    # previously failed models during retraining
                    original_use_trained = config_reader.get_property('models.use_trained', 'false')
                    config_reader.set_property('models.use_trained', 'false')
                    config_reader.save_properties()
                    
                    try:
                        # Fall back to training if loading fails
                        result = get_fraud_system(train_data_path=training_data_path)
                        
                        # Handle different return value counts from get_fraud_system
                        if isinstance(result, tuple) and len(result) == 3:
                            system, config_reader, evaluation_results = result
                        elif isinstance(result, tuple) and len(result) == 2:
                            system, config_reader = result
                            evaluation_results = {}  # Default empty evaluation results
                            logger.warning("No evaluation results returned from get_fraud_system")
                        else:
                            raise ValueError(f"Unexpected return format from get_fraud_system: {result}")
                        
                        # Get current timestamp
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        
                        # Update config with both timestamps
                        config_reader.set_property('models.timestamp', timestamp)
                        config_reader.set_property('models.last_data_timestamp', data_file_timestamp)
                        config_reader.set_property('models.use_trained', 'true')
                        config_reader.save_properties()
                        
                        logger.info(f"Model retrained and saved with timestamp {timestamp}")
                        logger.info(f"Training data timestamp recorded as {data_file_timestamp}")
                    except Exception as e:
                        # If retraining fails, restore original setting
                        logger.error(f"Retraining failed: {str(e)}")
                        config_reader.set_property('models.use_trained', original_use_trained)
                        config_reader.save_properties()
                        raise
        else:
            # No training data found
            logger.warning(f"Training data not found at {training_data_path}")
            system, config_reader = get_fraud_system()

        

        # Get the current directory as the default base path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        config_base_path = config_reader.get_property('base.path',current_dir)
    
        if config_base_path:
            base_path= Path(config_base_path)
        else:
            print("No base path..exiting..")
            sys.exit(1)

        incoming_folder = base_path / 'incoming_messages'
        processed_folder = base_path / 'processed_messages'
        failed_folder = base_path / 'failed_messages'
        
        # Create directories if they don't exist
        processed_folder.mkdir(parents=True, exist_ok=True)
        failed_folder.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Using directories:")
        logger.info(f"Incoming: {incoming_folder}")
        logger.info(f"Processed: {processed_folder}")
        logger.info(f"Failed: {failed_folder}")
        
        # Initialize multi-format processor
        processor = MultiFormatProcessor(system)
        
        # Enhance it with return processing capabilities
        processor = ReturnProcessorIntegration.integrate_with_processor(processor)


        # Process files (ACH, ACH Returns, XML, and CSV)
        results = processor.process_files(incoming_folder, processed_folder, failed_folder)
        # Process ACH return files
        #return_results = processor.process_return_files(incoming_folder, processed_folder, failed_folder)

        # Print summary
        logger.info("\nProcessing Summary:")
        logger.info(f"Total Messages: {results['total_processed'] + results['failed_processing']}")
        logger.info(f"Successfully Processed: {results['total_processed']}")
        logger.info(f"   - CSV Files: {results['csv_processed']}")
        logger.info(f"   - XML Files: {results['xml_processed']}")
        logger.info(f"   - ACH Files: {results['ach_processed']}")
        logger.info(f"   - Return Files: {results.get('return_processed', 0)}")
        logger.info(f"Return Processing:")
        logger.info(f"   - Returns Matched: {results.get('return_matched', 0)}")
        logger.info(f"   - Returns Unmatched: {results.get('return_unmatched', 0)}")
        logger.info(f"Alerts Generated: {results['alerts_generated']}")
        logger.info(f"Failed Processing: {results['failed_processing']}")

      
                
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)
    finally:
        if 'db' in locals():
            db.close()


if __name__ == "__main__":
    
    main()
