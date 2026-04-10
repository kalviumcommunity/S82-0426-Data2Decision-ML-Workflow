from typing import List, Dict

def get_recommendations(predicted_skills: List[str]) -> Dict[str, List[str]]:
    """
    Provides skill recommendations and project ideas based on detected skills.
    """
    recommendations = {
        "missing_skills": [],
        "suggested_next_projects": []
    }
    
    # 1. Tech Stack Logic
    if "React" in predicted_skills and "Flask" not in predicted_skills:
        recommendations["missing_skills"].append("Backend API (Flask/Node.js)")
        recommendations["suggested_next_projects"].append("Build a Full-Stack E-commerce app")
        
    if "Python" in predicted_skills and "Machine_Learning" not in predicted_skills:
        recommendations["missing_skills"].append("Data Science Libraries (Scikit-Learn/Pandas)")
        recommendations["suggested_next_projects"].append("Predictive analysis on stock prices")
        
    if "SQL" not in predicted_skills:
        recommendations["missing_skills"].append("Database Management (SQL/PostgreSQL)")
        recommendations["suggested_next_projects"].append("Inventory Management System with DB")
        
    if "Machine_Learning" in predicted_skills and "Flask" not in predicted_skills:
        recommendations["missing_skills"].append("Model Deployment (Flask/Docker)")
        recommendations["suggested_next_projects"].append("Deploy your ML model as a REST API")

    # Default if everything found or unknown
    if not recommendations["missing_skills"]:
        recommendations["missing_skills"].append("Cloud Deployment (AWS/Azure)")
        recommendations["suggested_next_projects"].append("Optimize for scale and high availability")

    return recommendations
