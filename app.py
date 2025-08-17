from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field, computed_field
import pickle
import pandas as pd
import logging
import os
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global variable to store the model
best_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global best_model
    try:
        model_path = os.getenv("MODEL_PATH", "student_model.pkl")
        logger.info(f"Loading model from {model_path}")
        with open(model_path, 'rb') as f:
            best_model = pickle.load(f)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Could not load model: {e}")
    
    yield
    
    # Shutdown
    logger.info("Application shutting down")

# Build the FastAPI object with metadata and lifespan
app = FastAPI(
    title="Student Performance Prediction API",
    description="API for predicting student performance based on quiz interaction data",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add security middleware (simplified for Render)
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["*"]  # Render handles this
)

# Add CORS middleware (simplified for Render)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Pydantic model to validate incoming data
class UserInput(BaseModel):
    hint_count: float = Field(..., ge=0, description='total hints used by the student during the Quiz till now')
    bottom_hint: float = Field(..., ge=0, description='Hint use by the student during the Quiz')
    attempt_count: float = Field(..., ge=0, description='Total attemps from starting to till now')
    ms_first_response: float = Field(..., ge=0, description='Time to attempt the question in first time')
    duration: float = Field(..., ge=0, description='total duration to solve the quiz')
    avg_conf_frustrated: float = Field(..., ge=0, le=1, description='frustrated')
    avg_conf_confused: float = Field(..., ge=0, le=1, description='confused')
    avg_conf_concentrating: float = Field(..., ge=0, le=1, description='concentrating')
    avg_conf_bored: float = Field(..., ge=0, le=1, description='bored')
    
    # Computed Features
    @computed_field
    @property
    def action_count(self) -> float:
        """Total actions = attempts + hints."""
        return self.attempt_count + self.hint_count
    
    @computed_field
    @property
    def action_level(self) -> int:
        """Categorical simplification of action count."""
        if self.action_count == 0:
            return 0
        elif self.action_count == 1:
            return 1
        return 2   
    
    @computed_field
    @property
    def hint_dependency(self) -> float:
        return self.hint_count / (self.attempt_count + 1)
    
    @computed_field
    @property
    def response_speed(self) -> float:
        return 1 / (self.ms_first_response + 1)
    
    @computed_field
    @property
    def confidence_balance(self) -> float:
        return self.avg_conf_concentrating - self.avg_conf_frustrated - self.avg_conf_confused
    
    @computed_field
    @property
    def engagement_ratio(self) -> float:
        return self.avg_conf_concentrating / (self.avg_conf_bored + 1e-6)  
    
    @computed_field
    @property
    def efficiency_indicator(self) -> float:
        return self.action_count / (self.attempt_count + 1)

print("Pydantic model defined")


# ---------------------------
# Performance Categorization Functions
# ---------------------------
def categorize_student_performance(correctness_score: float) -> tuple[int, str, str, str]:
    if correctness_score < 0.3:
        return (0, "Poor", "Needs immediate intervention and support", "🆘")
    elif correctness_score < 0.45:
        return (1, "Weak", "Requires additional practice and guidance", "⚠️")
    elif correctness_score < 0.6:
        return (2, "Below Average", "Shows potential but needs improvement", "📈")
    elif correctness_score < 0.75:
        return (3, "Average", "Solid understanding with room to grow", "✅")
    elif correctness_score < 0.9:
        return (4, "Strong", "Excellent performance and comprehension", "🌟")
    else:
        return (5, "Outstanding", "Exceptional mastery of the material", "🏆")


def recommend_learning_material(category_number: int) -> str:
    recommendations = {
        0: "🔹 Basics tutorial video + guided beginner-level exercises.",
        1: "🔸 Visual explanation content + step-by-step practice problems.",
        2: "🔹 Practice exercises with hints enabled + instant feedback.",
        3: "✅ Standard module content + end-of-lesson quiz.",
        4: "🌟 Advanced challenge problems + peer group discussion tasks.",
        5: "🏆 Project-based learning module + opportunity to mentor peers."
    }
    return recommendations.get(category_number, "📘 Keep learning and practicing regularly.")

def generate_feedback_message(category_number: int) -> str:
    feedback = {
        0: "It's okay to struggle — the key is to keep going. Let's review the basics together.",
        1: "You're making progress. Focus on the foundation, and don't hesitate to seek help.",
        2: "You've got potential. A little more consistent effort will go a long way!",
        3: "Nice work! You're on track — just refine your skills step by step.",
        4: "Great job! You've developed a solid understanding. Keep challenging yourself.",
        5: "Outstanding! You've truly mastered the topic. Consider exploring advanced material or helping peers."
    }
    return feedback.get(category_number, "Keep pushing forward — every step counts!")

def generate_learner_profile(data: UserInput) -> str:
    duration = data.duration
    attempt_count = data.attempt_count
    concentrating = data.avg_conf_concentrating
    frustrated = data.avg_conf_frustrated
    confused = data.avg_conf_confused
    bottom_hint = data.bottom_hint
    confidence_balance = data.confidence_balance
    efficiency = data.efficiency_indicator
    hint_count = data.hint_count
    hint_dependency = data.hint_dependency
   
    if (duration < 1800 and attempt_count < 3 and concentrating < 0.5 and frustrated > 0.3):
        return "Fast but Careless 🐇"
   
    if (duration > 1800 and hint_count < 5 and concentrating > 0.6 and efficiency > 0.6):
        return "Slow and Careful 🐢"
   
    if (hint_count > 6 and confused > 0.3 and bottom_hint > 5 and confidence_balance < 0.4):
        return "Confused Learner 🤔"
   
    if (concentrating > 0.6 and confidence_balance > 0.6 and hint_dependency < 0.3 and efficiency > 0.6):
        return "Focused Performer 🎯"
   
    return "General Learner"

# ---------------------------
# Helper functions to classify behaviors
# ---------------------------
def classify_engagement(ratio: float) -> str:
    if ratio > 2: return "High"
    elif ratio > 1: return "Moderate"
    return "Low"

def classify_hint_dependency(dep: float) -> str:
    if dep > 0.7: return "High"
    elif dep > 0.3: return "Moderate"
    return "Low"

def classify_efficiency(val: float) -> str:
    if val > 1: return "High"
    elif val > 0.5: return "Moderate"
    return "Low"

def classify_response_speed(speed: float) -> str:
    if speed > 0.05: return "Fast"
    elif speed > 0.01: return "Moderate"
    return "Slow"

def classify_confidence(balance: float) -> str:
    if balance > 0.2: return "Positive"
    elif balance > -0.2: return "Neutral"
    return "Negative"

def classify_persistence(attempts: float) -> str:
    if attempts > 5: return "High"
    elif attempts > 2: return "Moderate"
    return "Low"

def classify_activity(actions: float) -> str:
    if actions == 0: return "None"
    elif actions == 1: return "Minimal"
    return "Active"


# ---------------------------
# API Endpoints
# ---------------------------
@app.get("/")
def root():
    return {"message": "Student Performance Analysis API - Ready to analyze learning patterns!"}

@app.post('/predict')
def predicted_correctness(data: UserInput):
    # Prepare input data for ML model
    input_df = pd.DataFrame([{
        'hint_count': data.hint_count,
        'bottom_hint': data.bottom_hint,
        'attempt_count': data.attempt_count,
        'ms_first_response': data.ms_first_response,
        'duration': data.duration,
        'avg_conf_frustrated': data.avg_conf_frustrated,
        'avg_conf_confused': data.avg_conf_confused,
        'avg_conf_concentrating': data.avg_conf_concentrating,
        'avg_conf_bored': data.avg_conf_bored,
        'action_count': data.action_count,
        'hint_dependency': data.hint_dependency,
        'response_speed': data.response_speed,
        'confidence_balance': data.confidence_balance,
        'engagement_ratio': data.engagement_ratio,
        'efficiency_indicator': data.efficiency_indicator
    }])
    
    # Check if model is loaded
    if best_model is None:
        logger.error("Model not loaded")
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Get prediction
        predicted_correctness = best_model.predict(input_df)[0]
        
        # Get performance category
        category_num, category_name, description, emoji = categorize_student_performance(predicted_correctness)
        
        # Get learner profile
        learner_profile = generate_learner_profile(data)
        
        # Generate behavior analysis
        behaviors = {
            "engagement": classify_engagement(data.engagement_ratio),
            "hint_dependency": classify_hint_dependency(data.hint_dependency),
            "efficiency": classify_efficiency(data.efficiency_indicator),
            "response_speed": classify_response_speed(data.response_speed),
            "confidence_balance": classify_confidence(data.confidence_balance),
            "persistence": classify_persistence(data.attempt_count),
            "overall_activity": classify_activity(data.action_count)
        }
        
        # Get recommendations and feedback
        recommendation = recommend_learning_material(category_num)
        feedback = generate_feedback_message(category_num)
    
        # Prepare comprehensive response
        response = {
            "prediction": {
                "correctness_score": float(predicted_correctness),
                "category_number": category_num,
                "performance_category": category_name,
                "description": description,
                "emoji": emoji
            },
            "learner_profile": learner_profile,
            "behaviors": behaviors,
            "recommendations": {
                "learning_material": recommendation,
                "feedback_message": feedback
            },
            "raw_metrics": {
                "action_count": data.action_count,
                "hint_dependency": data.hint_dependency,
                "response_speed": data.response_speed,
                "confidence_balance": data.confidence_balance,
                "engagement_ratio": data.engagement_ratio,
                "efficiency_indicator": data.efficiency_indicator
            }
        }
        
        return JSONResponse(status_code=200, content=response)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post('/analyze')
def analyze_student_behavior(data: UserInput):
    """Endpoint for detailed behavioral analysis without ML prediction"""
    
    # Get learner profile
    learner_profile = generate_learner_profile(data)
    
    # Generate behavior analysis
    behaviors = {
        "engagement": classify_engagement(data.engagement_ratio),
        "hint_dependency": classify_hint_dependency(data.hint_dependency),
        "efficiency": classify_efficiency(data.efficiency_indicator),
        "response_speed": classify_response_speed(data.response_speed),
        "confidence_balance": classify_confidence(data.confidence_balance),
        "persistence": classify_persistence(data.attempt_count),
        "overall_activity": classify_activity(data.action_count)
    }
    
    response = {
        "learner_profile": learner_profile,
        "behaviors": behaviors,
        "computed_metrics": {
            "action_count": data.action_count,
            "hint_dependency": data.hint_dependency,
            "response_speed": data.response_speed,
            "confidence_balance": data.confidence_balance,
            "engagement_ratio": data.engagement_ratio,
            "efficiency_indicator": data.efficiency_indicator
        }
    }
    
    return JSONResponse(status_code=200, content=response)

@app.get('/health')
def health_check() -> dict[str, bool | str]:
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": best_model is not None}


print("Routes defined")
print("App setup complete!")