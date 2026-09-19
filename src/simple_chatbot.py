"""
Simple fallback implementation of the multilingual code-mix chatbot.
This module provides a basic chatbot that doesn't rely on external libraries.
"""
import re
from typing import Dict, List, Optional

class SimpleChatbot:
    """Simple implementation of a multilingual chatbot."""
    
    def __init__(self):
        """Initialize the chatbot."""
        self.responses = {
            "en": {
                "greeting": "Hello! How can I help you today?",
                "weather": "I'm sorry, I don't have access to weather information.",
                "booking": "I don't have information about your booking.",
                "restaurant": "I don't have information about restaurants.",
                "reminder": "I can't set reminders at the moment.",
                "unknown": "I'm sorry, I don't understand. Could you please rephrase your question?",
                "farewell": "Goodbye! Have a great day!"
            },
            "hi": {
                "greeting": "नमस्ते! मैं आपकी कैसे मदद कर सकता हूँ?",
                "weather": "मुझे खेद है, मेरे पास मौसम की जानकारी नहीं है।",
                "booking": "मुझे आपकी बुकिंग के बारे में जानकारी नहीं है।",
                "restaurant": "मुझे रेस्तरां के बारे में जानकारी नहीं है।",
                "reminder": "मैं अभी रिमाइंडर सेट नहीं कर सकता।",
                "unknown": "मुझे खेद है, मैं समझ नहीं पाया। कृपया अपना प्रश्न दोबारा पूछें।",
                "farewell": "अलविदा! आपका दिन शुभ हो!"
            },
            "bn": {
                "greeting": "হ্যালো! আমি আপনাকে কিভাবে সাহায্য করতে পারি?",
                "weather": "দুঃখিত, আমার কাছে আবহাওয়ার তথ্য নেই।",
                "booking": "আমার কাছে আপনার বুকিং সম্পর্কে কোন তথ্য নেই।",
                "restaurant": "আমার কাছে রেস্তোরাঁ সম্পর্কে কোন তথ্য নেই।",
                "reminder": "আমি এই মুহূর্তে রিমাইন্ডার সেট করতে পারি না।",
                "unknown": "দুঃখিত, আমি বুঝতে পারিনি। অনুগ্রহ করে আপনার প্রশ্ন পুনরায় জিজ্ঞাসা করুন।",
                "farewell": "বিদায়! আপনার দিন শুভ হোক!"
            }
        }
        
        # Keywords for intent detection
        self.keywords = {
            "weather": ["weather", "temperature", "rain", "forecast", "मौसम", "तापमान", "बारिश", "আবহাওয়া", "তাপমাত্রা", "বৃষ্টি"],
            "booking": ["booking", "reservation", "ticket", "बुकिंग", "आरक्षण", "टिकट", "বুকিং", "রিজার্ভেশন", "টিকেট"],
            "restaurant": ["restaurant", "food", "dinner", "lunch", "रेस्तरां", "खाना", "डिनर", "लंच", "রেস্তোরাঁ", "খাবার", "ডিনার", "লাঞ্চ"],
            "reminder": ["reminder", "remind", "alert", "रिमाइंडर", "याद", "अलर्ट", "রিমাইন্ডার", "মনে করিয়ে", "অ্যালার্ট"],
            "farewell": ["bye", "goodbye", "exit", "quit", "अलविदा", "बाय", "निकास", "बंद", "বিদায়", "বাই", "প্রস্থান", "বন্ধ"]
        }
    
    def detect_language(self, text: str) -> str:
        """
        Detect the primary language of the text.
        
        Args:
            text: Input text
            
        Returns:
            Language code ('en', 'hi', or 'bn')
        """
        # Check for Devanagari script (Hindi)
        if re.search(r'[\u0900-\u097F]', text):
            return "hi"
        
        # Check for Bengali script
        if re.search(r'[\u0980-\u09FF]', text):
            return "bn"
        
        # Default to English
        return "en"
    
    def detect_intent(self, text: str) -> str:
        """
        Detect the intent of the text.
        
        Args:
            text: Input text
            
        Returns:
            Intent ('greeting', 'weather', 'booking', 'restaurant', 'reminder', 'farewell', or 'unknown')
        """
        text = text.lower()
        
        # Check for greeting
        if any(word in text for word in ["hello", "hi", "hey", "नमस्ते", "हैलो", "হ্যালো", "হাই"]):
            return "greeting"
        
        # Check for other intents
        for intent, keywords in self.keywords.items():
            if any(keyword in text for keyword in keywords):
                return intent
        
        # Default to unknown
        return "unknown"
    
    def generate_response(self, text: str) -> str:
        """
        Generate a response to the input text.
        
        Args:
            text: Input text
            
        Returns:
            Response text
        """
        # Detect language and intent
        language = self.detect_language(text)
        intent = self.detect_intent(text)
        
        # Get response
        return self.responses[language][intent]
    
    def chat(self):
        """Run an interactive chat session."""
        print("\n===== Simple Multilingual Code-Mix Chatbot =====")
        print("Type 'exit' or 'quit' to end the conversation.")
        print("=========================================\n")
        
        while True:
            # Get user input
            user_input = input("\nYou: ")
            
            # Check for exit command
            if user_input.lower() in ["exit", "quit", "bye", "goodbye"]:
                print("Chatbot: Goodbye!")
                break
            
            # Generate response
            response = self.generate_response(user_input)
            print(f"Chatbot: {response}")

def run_simple_chatbot():
    """Run the simple chatbot."""
    chatbot = SimpleChatbot()
    chatbot.chat()
