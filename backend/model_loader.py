"""
Backend module for loading the emotion detection model and making predictions.
"""
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import os


class EmotionDetector:
    """Class to handle emotion detection model loading and predictions."""
    
    def __init__(self, model_path="final_emotion_model"):
        """
        Initialize the emotion detector with the model path.
        
        Args:
            model_path (str): Path to the trained model directory
        """
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.id2label = {
            0: "sadness",
            1: "joy",
            2: "love",
            3: "anger",
            4: "fear",
            5: "surprise"
        }
        
    def load_model(self):
        """Load the pre-trained model and tokenizer."""
        try:
            print(f"Loading model from {self.model_path}...")
            self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_path)
            self.model = DistilBertForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def predict(self, text):
        """
        Predict the emotion from the input text.
        
        Args:
            text (str): Input sentence to analyze
            
        Returns:
            dict: Dictionary containing predicted emotion and confidence scores
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Please call load_model() first.")
        
        # Tokenize the input text
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Move inputs to the same device as the model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
        
        # Get the predicted class and confidence
        predicted_id = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0][predicted_id].item()
        
        # Get all emotion scores
        emotion_scores = {}
        for emotion_id, emotion_name in self.id2label.items():
            emotion_scores[emotion_name] = probabilities[0][emotion_id].item()
        
        # Sort emotions by confidence
        sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "predicted_emotion": self.id2label[predicted_id],
            "confidence": confidence,
            "all_emotions": emotion_scores,
            "sorted_emotions": sorted_emotions
        }

