import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
from collections import defaultdict
import os
import joblib
import json


class GaitValidator:
    """Validate and test the performance of gait identification system"""
    
    def __init__(self, gait_analyzer=None, data_path=None):
        """Initialize the validator with optional gait analyzer and data path"""
        self.gait_analyzer = gait_analyzer
        self.data_path = data_path
        self.features_df = None
        self.known_identities = {}  # Map track_ids to actual person identities
        self.feature_importance = {}
        self.classifier = None
        self.scaler = StandardScaler()
        self.pca = None
        self.test_results = {}
        
    def load_features_from_csv(self, csv_path):
        """Load feature data from CSV file"""
        if not os.path.exists(csv_path):
            print(f"Error: File {csv_path} not found")
            return False
            
        self.features_df = pd.read_csv(csv_path)
        print(f"Loaded {len(self.features_df)} feature records from {csv_path}")
        return True
    
    def assign_identity(self, track_id, person_name):
        """Manually assign identity to a track_id"""
        self.known_identities[track_id] = person_name
        return True
    
    def load_identities_from_json(self, json_path):
        """Load track_id to person identity mappings from JSON file"""
        try:
            with open(json_path, 'r') as f:
                self.known_identities = json.load(f)
                # Convert string keys to integers if needed
                self.known_identities = {int(k): v for k, v in self.known_identities.items()}
            return True
        except Exception as e:
            print(f"Error loading identities: {e}")
            return False
    
    def save_identities_to_json(self, json_path):
        """Save track_id to person identity mappings to JSON file"""
        try:
            with open(json_path, 'w') as f:
                json.dump(self.known_identities, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving identities: {e}")
            return False
    
    def analyze_feature_quality(self):
        """Analyze the quality and distinctiveness of extracted features"""
        if self.features_df is None or len(self.features_df) == 0:
            print("No feature data available to analyze")
            return {}
        
        results = {}
        
        # 1. Check for missing values
        missing_counts = self.features_df.isna().sum()
        missing_pct = (missing_counts / len(self.features_df)) * 100
        results['missing_values'] = {
            'counts': missing_counts.to_dict(),
            'percentages': missing_pct.to_dict()
        }
        
        # 2. Calculate feature variability (features with higher CV are more distinctive)
        # Exclude non-numeric and ID columns
        feature_cols = self.features_df.select_dtypes(include=[np.number]).columns.tolist()
        if 'track_id' in feature_cols:
            feature_cols.remove('track_id')
            
        if not feature_cols:
            print("No numeric feature columns found")
            return results
            
        # Calculate coefficient of variation (CV = std/mean)
        means = self.features_df[feature_cols].mean()
        stds = self.features_df[feature_cols].std()
        cv = stds / means.abs()
        
        # Replace infinities with NaN
        cv = cv.replace([np.inf, -np.inf], np.nan)
        
        # Sort features by variability
        cv = cv.dropna().sort_values(ascending=False)
        results['feature_variability'] = cv.to_dict()
        
        # 3. Calculate intra-person vs inter-person variability (if identities are known)
        if self.known_identities and len(self.known_identities) > 1:
            # Create a version of the dataframe with person identities
            df_with_identity = self.features_df.copy()
            df_with_identity['person_identity'] = df_with_identity['track_id'].map(
                lambda x: self.known_identities.get(x, f"Unknown-{x}")
            )
            
            # Calculate intra-person variability (within same person)
            intra_var = {}
            inter_var = {}
            
            for feature in feature_cols:
                # Skip if feature has too many missing values
                if missing_pct[feature] > 50:
                    continue
                    
                # Calculate intra-person variability (average of std for each person)
                intra_person_std = df_with_identity.groupby('person_identity')[feature].std().mean()
                intra_var[feature] = intra_person_std
                
                # Calculate inter-person variability (std of means across people)
                inter_person_std = df_with_identity.groupby('person_identity')[feature].mean().std()
                inter_var[feature] = inter_person_std
            
            # Calculate ratio of inter to intra variability
            # Higher ratio means feature is better for discrimination
            feature_quality = {}
            for feature in intra_var:
                if intra_var[feature] > 0:
                    ratio = inter_var[feature] / intra_var[feature]
                    feature_quality[feature] = ratio
            
            # Sort by quality ratio
            feature_quality = {k: v for k, v in sorted(feature_quality.items(), 
                                                      key=lambda item: item[1], reverse=True)}
            
            results['feature_quality'] = feature_quality
            self.feature_importance = feature_quality
        
        return results
    
    def visualize_feature_importance(self, top_n=10, save_path=None):
        """Visualize feature importance for identification"""
        if not self.feature_importance:
            print("No feature importance data available. Run analyze_feature_quality first.")
            return
        
        # Get top N features
        top_features = dict(list(self.feature_importance.items())[:top_n])
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(top_features.keys(), top_features.values())
        plt.xticks(rotation=45, ha='right')
        plt.title('Feature Importance for Person Identification')
        plt.xlabel('Feature')
        plt.ylabel('Discrimination Power (Inter/Intra Variability)')
        plt.tight_layout()
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', rotation=0)
        
        if save_path:
            plt.savefig(save_path)
            print(f"Feature importance plot saved to {save_path}")
            
        plt.show()
    
    def split_train_test(self, test_size=0.3, random_state=42):
        """Split data into training and testing sets"""
        if self.features_df is None or len(self.features_df) == 0:
            print("No feature data available")
            return None, None
            
        if not self.known_identities:
            print("No identity information available. Using track_ids as identities.")
            self.features_df['person_identity'] = self.features_df['track_id']
        else:
            self.features_df['person_identity'] = self.features_df['track_id'].map(
                lambda x: self.known_identities.get(x, f"Unknown-{x}")
            )
            
        # Select features and target
        features = self.features_df.select_dtypes(include=[np.number]).drop(['track_id'], axis=1, errors='ignore')
        target = self.features_df['person_identity']
        
        # Handle missing values
        features = features.fillna(0)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=test_size, random_state=random_state
        )
        
        print(f"Split data into {len(X_train)} training and {len(X_test)} testing samples")
        return (X_train, y_train), (X_test, y_test)
    
    def train_classifier(self, n_neighbors=3, weights='distance'):
        """Train a classifier on the feature data"""
        if self.features_df is None or len(self.features_df) == 0:
            print("No feature data available")
            return False
            
        # Split the data
        (X_train, y_train), (X_test, y_test) = self.split_train_test()
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Optionally apply PCA if we have many features
        if X_train.shape[1] > 15:
            self.pca = PCA(n_components=0.95)  # Keep 95% of variance
            X_train_scaled = self.pca.fit_transform(X_train_scaled)
            print(f"Applied PCA, reduced features from {X_train.shape[1]} to {X_train_scaled.shape[1]}")
        
        # Train classifier
        self.classifier = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
        self.classifier.fit(X_train_scaled, y_train)
        
        # Test on training data
        train_predictions = self.classifier.predict(X_train_scaled)
        train_accuracy = accuracy_score(y_train, train_predictions)
        
        print(f"Training accuracy: {train_accuracy:.4f}")
        return True
    
    def evaluate_classifier(self):
        """Evaluate classifier performance on test data"""
        if self.classifier is None:
            print("No trained classifier available")
            return None
        
        # Split the data if not already done
        (X_train, y_train), (X_test, y_test) = self.split_train_test()
        
        # Scale and transform test data
        X_test_scaled = self.scaler.transform(X_test)
        if self.pca is not None:
            X_test_scaled = self.pca.transform(X_test_scaled)
        
        # Make predictions
        predictions = self.classifier.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, predictions, average='weighted'
        )
        
        # Create confusion matrix
        cm = confusion_matrix(y_test, predictions)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'classes': self.classifier.classes_
        }
        
        self.test_results = results
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        return results
    
    def visualize_confusion_matrix(self, save_path=None):
        """Visualize confusion matrix of test results"""
        if not self.test_results or 'confusion_matrix' not in self.test_results:
            print("No test results available. Run evaluate_classifier first.")
            return
        
        cm = self.test_results['confusion_matrix']
        classes = self.test_results['classes']
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Confusion matrix saved to {save_path}")
            
        plt.show()
    
    def save_model(self, model_path="gait_classifier_model.pkl"):
        """Save the trained classifier and preprocessors"""
        if self.classifier is None:
            print("No trained classifier to save")
            return False
        
        model_dict = {
            'classifier': self.classifier,
            'scaler': self.scaler,
            'pca': self.pca,
            'feature_importance': self.feature_importance
        }
        
        try:
            joblib.dump(model_dict, model_path)
            print(f"Model saved to {model_path}")
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def load_model(self, model_path="gait_classifier_model.pkl"):
        """Load a previously trained classifier"""
        try:
            model_dict = joblib.load(model_path)
            
            self.classifier = model_dict['classifier']
            self.scaler = model_dict['scaler']
            self.pca = model_dict['pca']
            self.feature_importance = model_dict['feature_importance']
            
            print(f"Model loaded from {model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def identify_person(self, feature_vector, confidence_threshold=0.7):
        """Identify a person based on gait features"""
        if self.classifier is None:
            print("No trained classifier available")
            return None, 0.0
        
        # Scale and transform feature vector
        feature_vector = np.array(feature_vector).reshape(1, -1)
        
        # Handle feature count mismatch
        expected_features = self.scaler.n_features_in_
        actual_features = feature_vector.shape[1]
        
        if actual_features != expected_features:
            print(f"Feature count mismatch: got {actual_features}, expected {expected_features}")
            
            # Create properly sized feature vector
            if actual_features < expected_features:
                # Pad with zeros if we have too few features
                padded_vector = np.zeros((1, expected_features))
                padded_vector[0, :actual_features] = feature_vector[0]
                feature_vector = padded_vector
            else:
                # Truncate if we have too many features
                feature_vector = feature_vector[:, :expected_features]
        
        try:
            # Apply transformations
            feature_scaled = self.scaler.transform(feature_vector)
            
            if self.pca is not None:
                feature_scaled = self.pca.transform(feature_scaled)
            
            # Get prediction and probabilities
            prediction = self.classifier.predict(feature_scaled)[0]
            probabilities = self.classifier.predict_proba(feature_scaled)[0]
            confidence = np.max(probabilities)
            
            if confidence < confidence_threshold:
                return "Unknown", confidence
            
            return prediction, confidence
        except Exception as e:
            print(f"Error during identification: {str(e)}")
            return "Error", 0.0