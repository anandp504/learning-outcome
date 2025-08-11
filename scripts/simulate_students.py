#!/usr/bin/env python3
"""Simulate student performance data for testing the learning recommendation system."""

import json
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any


def generate_student_profile(student_id: str, name: str, grade_level: int = 6) -> Dict[str, Any]:
    """Generate a realistic student profile."""
    
    # Learning style preferences
    learning_styles = ["visual", "auditory", "kinesthetic", "reading_writing"]
    math_aptitudes = ["struggling", "below_average", "average", "above_average", "excellent"]
    
    # Interests that can be related to math
    math_interests = [
        "puzzles", "games", "sports", "music", "art", "cooking", "building", "nature",
        "technology", "space", "animals", "cars", "robots", "magic", "treasure_hunting"
    ]
    
    # Learning goals
    learning_goals = [
        "Improve problem-solving skills",
        "Master basic math operations",
        "Prepare for algebra",
        "Build confidence in math",
        "Learn practical math applications",
        "Develop logical thinking"
    ]
    
    student = {
        "student_id": student_id,
        "name": name,
        "grade_level": grade_level,
        "age": random.randint(10, 12),
        "learning_style": random.choice(learning_styles),
        "math_aptitude": random.choice(math_aptitudes),
        "interests": random.sample(math_interests, random.randint(2, 4)),
        "goals": random.sample(learning_goals, random.randint(2, 3)),
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }
    
    return student


def generate_performance_data(student_id: str, concept_id: str, 
                            difficulty: int, base_performance: float) -> Dict[str, Any]:
    """Generate realistic performance data for a concept."""
    
    # Adjust performance based on difficulty and base performance
    difficulty_factor = 1.0 - (difficulty - 1) * 0.1  # Higher difficulty = lower performance
    adjusted_performance = base_performance * difficulty_factor
    
    # Add some randomness
    performance_variation = random.uniform(-0.2, 0.2)
    final_performance = max(0.0, min(1.0, adjusted_performance + performance_variation))
    
    # Generate realistic time spent based on performance and difficulty
    base_time = difficulty * 1.5  # Base time in hours
    if final_performance < 0.5:
        time_multiplier = random.uniform(1.5, 2.5)  # Struggling students spend more time
    elif final_performance > 0.8:
        time_multiplier = random.uniform(0.7, 1.2)  # High performers spend less time
    else:
        time_multiplier = random.uniform(1.0, 1.5)
    
    time_spent = base_time * time_multiplier
    
    # Generate attempts based on performance
    if final_performance < 0.4:
        attempts = random.randint(3, 6)
    elif final_performance < 0.7:
        attempts = random.randint(2, 4)
    else:
        attempts = random.randint(1, 3)
    
    # Generate strengths and weaknesses
    all_strengths = [
        "multiplication", "division", "addition", "subtraction", "number_sense",
        "problem_solving", "visual_representation", "logical_thinking", "patience",
        "attention_to_detail", "pattern_recognition", "estimation"
    ]
    
    all_weaknesses = [
        "word_problems", "large_numbers", "fractions", "decimals", "time_management",
        "test_anxiety", "reading_comprehension", "multi_step_problems", "mental_math",
        "concentration", "organization", "speed"
    ]
    
    # Select strengths and weaknesses based on performance
    num_strengths = max(1, int(final_performance * 4))
    num_weaknesses = max(1, int((1 - final_performance) * 4))
    
    strengths = random.sample(all_strengths, min(num_strengths, len(all_strengths)))
    weaknesses = random.sample(all_weaknesses, min(num_weaknesses, len(all_weaknesses)))
    
    # Determine mastery level
    if final_performance >= 0.9:
        mastery_level = "mastered"
    elif final_performance >= 0.7:
        mastery_level = "proficient"
    elif final_performance >= 0.5:
        mastery_level = "developing"
    else:
        mastery_level = "beginning"
    
    # Generate feedback
    feedback_options = {
        "mastered": [
            "Excellent work! You've mastered this concept completely.",
            "Outstanding performance! You're ready for more challenging material.",
            "Perfect! You demonstrate deep understanding of this topic."
        ],
        "proficient": [
            "Great job! You have a solid understanding of this concept.",
            "Well done! You're making excellent progress.",
            "Good work! You're ready to move to the next level."
        ],
        "developing": [
            "Good effort! Keep practicing to strengthen your understanding.",
            "You're on the right track! A bit more practice will help.",
            "Nice work! Focus on the areas that need improvement."
        ],
        "beginning": [
            "Keep trying! Math takes practice and patience.",
            "Don't give up! Let's work on this together.",
            "You can do this! Let's break it down into smaller steps."
        ]
    }
    
    feedback = random.choice(feedback_options[mastery_level])
    
    # Generate timestamp (within last 30 days)
    days_ago = random.randint(0, 30)
    last_attempt = datetime.now() - timedelta(days=days_ago)
    
    performance_data = {
        "student_id": student_id,
        "concept_id": concept_id,
        "performance_score": round(final_performance, 3),
        "attempts": attempts,
        "time_spent": round(time_spent, 2),
        "strengths": strengths,
        "weaknesses": weaknesses,
        "last_attempt": last_attempt.isoformat(),
        "mastery_level": mastery_level,
        "feedback": feedback
    }
    
    return performance_data


def generate_learning_journey(student_id: str, concepts: List[Dict[str, Any]], 
                             base_performance: float) -> Dict[str, Any]:
    """Generate a learning journey for a student."""
    
    # Select concepts for this student's journey
    num_concepts = random.randint(3, min(8, len(concepts)))
    selected_concepts = random.sample(concepts, num_concepts)
    
    # Create learning path
    learning_path = []
    completed_concepts = []
    in_progress_concepts = []
    
    for i, concept in enumerate(selected_concepts):
        if i < len(selected_concepts) - 1:
            completed_concepts.append(concept["concept_id"])
            learning_path.append(concept["concept_id"])
        else:
            in_progress_concepts.append(concept["concept_id"])
            learning_path.append(concept["concept_id"])
    
    # Calculate overall progress
    overall_progress = len(completed_concepts) / len(selected_concepts)
    
    # Calculate total study time
    total_study_time = sum(concept.get("estimated_hours", 3.0) for concept in selected_concepts)
    
    # Calculate average performance
    performance_scores = []
    for concept in selected_concepts:
        if concept["concept_id"] in completed_concepts:
            # Completed concepts have higher performance
            score = base_performance + random.uniform(0.1, 0.3)
            performance_scores.append(min(1.0, score))
        else:
            # In-progress concepts have lower performance
            score = base_performance - random.uniform(0.1, 0.3)
            performance_scores.append(max(0.0, score))
    
    average_performance = sum(performance_scores) / len(performance_scores)
    
    # Determine learning pace
    if total_study_time < 15:
        learning_pace = "fast"
    elif total_study_time > 25:
        learning_pace = "slow"
    else:
        learning_pace = "average"
    
    # Determine preferred difficulty
    if average_performance > 0.8:
        preferred_difficulty = random.randint(3, 5)
    elif average_performance > 0.6:
        preferred_difficulty = random.randint(2, 4)
    else:
        preferred_difficulty = random.randint(1, 3)
    
    # Generate next recommendations
    next_recommendations = []
    for concept in concepts:
        if concept["concept_id"] not in learning_path:
            # Check if prerequisites are met
            prereqs_met = all(prereq in completed_concepts for prereq in concept.get("prerequisites", []))
            if prereqs_met and len(next_recommendations) < 3:
                next_recommendations.append(concept["concept_id"])
    
    # Generate alternative paths
    alternative_paths = []
    if len(concepts) > len(learning_path) + 3:
        remaining_concepts = [c for c in concepts if c["concept_id"] not in learning_path]
        alt_path = random.sample(remaining_concepts, min(3, len(remaining_concepts)))
        alternative_paths.append([c["concept_id"] for c in alt_path])
    
    # Learning goals
    learning_goals = [
        "Complete current concept with 80% or higher",
        "Practice problem-solving skills",
        "Build confidence in math",
        "Prepare for next grade level"
    ]
    
    # Estimate completion
    if in_progress_concepts:
        current_concept = in_progress_concepts[0]
        current_concept_data = next(c for c in concepts if c["concept_id"] == current_concept)
        estimated_hours = current_concept_data.get("estimated_hours", 3.0)
        days_to_complete = int(estimated_hours / 0.5)  # Assuming 30 minutes per day
        estimated_completion = datetime.now() + timedelta(days=days_to_complete)
    else:
        estimated_completion = None
    
    learning_journey = {
        "journey_id": f"journey_{student_id}",
        "student_id": student_id,
        "current_concept": in_progress_concepts[0] if in_progress_concepts else None,
        "learning_path": learning_path,
        "completed_concepts": completed_concepts,
        "next_recommendations": next_recommendations,
        "alternative_paths": alternative_paths,
        "learning_goals": random.sample(learning_goals, random.randint(2, 3)),
        "estimated_completion": estimated_completion.isoformat() if estimated_completion else None,
        "overall_progress": round(overall_progress, 3),
        "total_study_time": round(total_study_time, 2),
        "average_performance": round(average_performance, 3),
        "learning_pace": learning_pace,
        "preferred_difficulty": preferred_difficulty,
        "last_activity": datetime.now().isoformat(),
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }
    
    return learning_journey


def generate_sample_students(num_students: int = 10) -> List[Dict[str, Any]]:
    """Generate a list of sample students with profiles and performance data."""
    
    # Sample student names
    first_names = [
        "Emma", "Liam", "Olivia", "Noah", "Ava", "Ethan", "Isabella", "Lucas",
        "Sophia", "Mason", "Mia", "Oliver", "Charlotte", "Elijah", "Amelia",
        "James", "Harper", "Benjamin", "Evelyn", "Sebastian"
    ]
    
    last_names = [
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
        "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
        "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin"
    ]
    
    students = []
    
    for i in range(num_students):
        # Generate student profile
        first_name = random.choice(first_names)
        last_name = random.choice(last_names)
        name = f"{first_name} {last_name}"
        student_id = f"student_{i+1:03d}"
        
        # Generate base performance level
        base_performance = random.uniform(0.3, 0.9)
        
        student_profile = generate_student_profile(student_id, name, 6)
        students.append(student_profile)
    
    return students


def generate_performance_dataset(students: List[Dict[str, Any]], 
                               concepts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate performance dataset for all students and concepts."""
    
    performance_data = []
    
    for student in students:
        # Generate base performance for this student
        base_performance = random.uniform(0.3, 0.9)
        
        # Generate performance for some concepts (not all)
        num_concepts_to_study = random.randint(3, min(8, len(concepts)))
        selected_concepts = random.sample(concepts, num_concepts_to_study)
        
        for concept in selected_concepts:
            performance = generate_performance_data(
                student["student_id"], 
                concept["concept_id"],
                concept["difficulty"],
                base_performance
            )
            performance_data.append(performance)
    
    return performance_data


def generate_learning_journeys(students: List[Dict[str, Any]], 
                              concepts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate learning journeys for all students."""
    
    learning_journeys = []
    
    for student in students:
        # Generate base performance for this student
        base_performance = random.uniform(0.3, 0.9)
        
        journey = generate_learning_journey(student["student_id"], concepts, base_performance)
        learning_journeys.append(journey)
    
    return learning_journeys


def main():
    """Main function to generate sample student data."""
    print("Generating Sample Student Data...")
    
    try:
        # Load concepts from knowledge graph
        with open("data/knowledge_graph.json", "r") as f:
            knowledge_graph = json.load(f)
        
        concepts = knowledge_graph["concepts"]
        print(f"📚 Loaded {len(concepts)} concepts from knowledge graph")
        
        # Generate students
        num_students = 15
        students = generate_sample_students(num_students)
        print(f"👥 Generated {len(students)} student profiles")
        
        # Generate performance data
        performance_data = generate_performance_dataset(students, concepts)
        print(f"📊 Generated {len(performance_data)} performance records")
        
        # Generate learning journeys
        learning_journeys = generate_learning_journeys(students, concepts)
        print(f"🛤️ Generated {len(learning_journeys)} learning journeys")
        
        # Save student profiles
        students_file = "data/sample_students.json"
        with open(students_file, "w") as f:
            json.dump({
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "total_students": len(students),
                    "total_performance_records": len(performance_data),
                    "total_learning_journeys": len(learning_journeys)
                },
                "students": students,
                "performance_data": performance_data,
                "learning_journeys": learning_journeys
            }, f, indent=2)
        
        print(f"✅ Sample student data generated successfully!")
        print(f"📁 Saved to: {students_file}")
        
        # Print summary statistics
        print("\n📊 Summary Statistics:")
        print(f"  • Total Students: {len(students)}")
        print(f"  • Total Performance Records: {len(performance_data)}")
        print(f"  • Total Learning Journeys: {len(learning_journeys)}")
        print(f"  • Average Performance Records per Student: {len(performance_data) / len(students):.1f}")
        
        # Print student performance distribution
        performance_scores = [p["performance_score"] for p in performance_data]
        avg_performance = sum(performance_scores) / len(performance_scores)
        print(f"  • Average Performance Score: {avg_performance:.3f}")
        
        # Print learning journey progress distribution
        progress_scores = [j["overall_progress"] for j in learning_journeys]
        avg_progress = sum(progress_scores) / len(progress_scores)
        print(f"  • Average Learning Progress: {avg_progress:.3f}")
        
        # Print sample student names
        print("\n👥 Sample Students:")
        for student in students[:5]:
            print(f"  • {student['name']} (ID: {student['student_id']})")
        if len(students) > 5:
            print(f"  • ... and {len(students) - 5} more")
        
    except FileNotFoundError:
        print("❌ Knowledge graph file not found. Please run generate_knowledge_graph.py first.")
    except Exception as e:
        print(f"❌ Error generating sample student data: {e}")
        raise


if __name__ == "__main__":
    main()
