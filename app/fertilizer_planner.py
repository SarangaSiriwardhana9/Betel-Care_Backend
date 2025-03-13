import datetime
from datetime import datetime, timedelta
from app.fertilizing_model import predict_7day_fertilizing_suitability

# Define fertilizer rotation and waiting periods (in days)
FERTILIZER_ROTATION = {
    "Gliricidia leaves": {
        "next": "Cow dung",
        "wait_days": 60  # 2 months
    },
    "Cow dung": {
        "next": "NPK (10-10-10)",
        "wait_days": 90  # 3 months
    },
    "NPK (10-10-10)": {
        "next": "Gliricidia leaves",
        "wait_days": 120  # 4 months
    }
}

# Default fertilizer to start with if no history is available
DEFAULT_FERTILIZER = "Gliricidia leaves"

def parse_fertilizer_history(history):
    """
    Parse the fertilizer history into a structured format
    
    Args:
        history (list): List of dictionaries with date and fertilizer info
            Example: [
                {"date": "2024-10-01", "fertilizer": "Gliricidia leaves"},
                {"date": "2024-12-05", "fertilizer": "Cow dung"}
            ]
    
    Returns:
        list: Sorted list of fertilizer applications by date (newest first)
    """
    parsed_history = []
    
    for entry in history:
        try:
            # Parse date (handle multiple formats)
            date_str = entry.get("date", "")
            fertilizer = entry.get("fertilizer", "")
            
            # Skip entries with missing data
            if not date_str or not fertilizer:
                continue
            
            # Try different date formats
            date_obj = None
            for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y"):
                try:
                    date_obj = datetime.strptime(date_str, fmt)
                    break
                except ValueError:
                    continue
            
            if date_obj:
                parsed_history.append({
                    "date": date_obj,
                    "fertilizer": fertilizer
                })
        except Exception as e:
            print(f"Error parsing fertilizer history entry: {e}")
    
    # Sort by date (newest first)
    return sorted(parsed_history, key=lambda x: x["date"], reverse=True)

def get_next_fertilizer_recommendation(history, today_date=None, location=None, rainfall_forecast=None):
    """
    Get recommendation for the next fertilizer application
    
    Args:
        history (list): List of fertilizer application history
        today_date (datetime, optional): Current date
        location (str, optional): Location for weather forecast
        rainfall_forecast (list, optional): 7-day rainfall forecast
        
    Returns:
        dict: Recommendation including next fertilizer, date, and weather suitability
    """
    if today_date is None:
        today_date = datetime.now()
    
    # Check if it's after 6 PM
    is_after_six_pm = today_date.hour >= 18
    
    # If it's after 6 PM, use tomorrow as the effective date for recommendations
    effective_date = today_date + timedelta(days=1) if is_after_six_pm else today_date
    
    # Parse history
    parsed_history = parse_fertilizer_history(history)
    
    # Initialize response
    response = {
        "last_application": None,
        "next_fertilizer": None,
        "recommended_date": None,
        "date_in_forecast": False,
        "date_has_passed": False,
        "weather_suitable": False,
        "weather_forecast": None,
        "alternative_date": None,
        "alternative_date_suitable": False,
        "message": "",
        "is_first_time": len(parsed_history) == 0,
        "is_after_six_pm": is_after_six_pm
    }
    
    # If history is empty, recommend the default fertilizer with immediate application
    if not parsed_history:
        response["next_fertilizer"] = DEFAULT_FERTILIZER
        response["recommended_date"] = effective_date.strftime("%Y-%m-%d")
        response["date_in_forecast"] = True
        
        if is_after_six_pm:
            response["message"] = "පොහොර යෙදීම හෙටින් ආරම්භ කරන්න"  # Start fertilizing from tomorrow
        else:
            response["message"] = "පොහොර යෙදීම අදින් ආරම්භ කරන්න"  # Start fertilizing from today
    else:
        # Get the most recent fertilizer application
        last_application = parsed_history[0]
        response["last_application"] = {
            "date": last_application["date"].strftime("%Y-%m-%d"),
            "fertilizer": last_application["fertilizer"]
        }
        
        # Find the next fertilizer in the rotation
        last_fertilizer = last_application["fertilizer"]
        
        # Handle case where the last fertilizer is not in our rotation chart
        if last_fertilizer not in FERTILIZER_ROTATION:
            next_fertilizer = DEFAULT_FERTILIZER
            wait_days = 30  # Default wait time for unknown fertilizers
            response["message"] = f"Unknown last fertilizer: {last_fertilizer}. Defaulting to {next_fertilizer} after 30 days."
        else:
            # Get next fertilizer and wait period from rotation chart
            next_fertilizer = FERTILIZER_ROTATION[last_fertilizer]["next"]
            wait_days = FERTILIZER_ROTATION[last_fertilizer]["wait_days"]
        
        # Calculate the recommended date for the next application
        recommended_date = last_application["date"] + timedelta(days=wait_days)
        response["next_fertilizer"] = next_fertilizer
        response["recommended_date"] = recommended_date.strftime("%Y-%m-%d")
        
        # Check if recommended date is today and it's after 6 PM
        if recommended_date.date() == today_date.date() and is_after_six_pm:
            # Move to tomorrow
            recommended_date = recommended_date + timedelta(days=1)
            response["recommended_date"] = recommended_date.strftime("%Y-%m-%d")
            response["message"] = f"Next fertilization with {next_fertilizer} recommended tomorrow."
        # Check if recommended date has already passed
        elif recommended_date < effective_date:
            response["date_has_passed"] = True
            response["message"] = f"Recommended date ({response['recommended_date']}) has already passed. Consider applying {next_fertilizer} soon."
        else:
            days_until = (recommended_date - effective_date).days
            response["message"] = f"Next fertilization with {next_fertilizer} recommended in {days_until} days."
    
    # If we have location and rainfall forecast, check weather suitability
    if location and rainfall_forecast and len(rainfall_forecast) > 0:
        # Get weather suitability for the next 7 days
        weather_forecast = predict_7day_fertilizing_suitability(location, rainfall_forecast)
        
        # Check if the weather forecast contains an error
        if len(weather_forecast) == 1 and 'error' in weather_forecast[0]:
            print(f"Weather forecast error: {weather_forecast[0]['error']}")
            response["message"] += " Could not get weather suitability information due to an error."
            response["weather_forecast"] = weather_forecast
        else:
            response["weather_forecast"] = weather_forecast
            
            # Check if the recommended date is within the forecast period
            try:
                # Format the recommended date for comparison
                recommended_date_str = response["recommended_date"]
                recommended_date_obj = datetime.strptime(recommended_date_str, "%Y-%m-%d")
                
                # Check if recommended date is within the forecast period
                forecast_dates = [day["date"] for day in weather_forecast]
                
                if recommended_date_str in forecast_dates:
                    response["date_in_forecast"] = True
                    
                    # Find the forecast for the recommended date
                    for day in weather_forecast:
                        if day["date"] == recommended_date_str:
                            response["weather_suitable"] = day["suitable_for_fertilizing"]
                            
                            if response["weather_suitable"]:
                                response["message"] += f" Weather on {recommended_date_str} is suitable for fertilizing."
                            else:
                                response["message"] += f" Weather on {recommended_date_str} is NOT suitable for fertilizing."
                            break
                
                # If date has passed or is not suitable, find an alternative date
                if response["date_has_passed"] or (response["date_in_forecast"] and not response["weather_suitable"]):
                    # Find suitable days in the forecast
                    suitable_days = [day for day in weather_forecast if day.get("suitable_for_fertilizing", False)]
                    
                    if suitable_days:
                        # Find the best day
                        best_day = max(suitable_days, key=lambda x: x.get("confidence", 0))
                        response["alternative_date"] = best_day["date"]
                        response["alternative_date_suitable"] = True
                        
                        if response["date_has_passed"]:
                            response["message"] += f" Recommend applying {next_fertilizer} on {best_day['date']} which has favorable weather."
                        else:
                            response["message"] += f" Consider postponing to {best_day['date']} for better weather conditions."
                    else:
                        response["message"] += " No suitable days found in the current weather forecast. Consider waiting for better conditions."
                
                # If recommended date is beyond the forecast period
                if not response["date_in_forecast"] and not response["date_has_passed"]:
                    response["message"] += f" The recommended date is beyond the current 7-day forecast period."
                    
            except Exception as e:
                print(f"Error processing weather forecast: {e}")
                response["message"] += " Could not process weather suitability information due to an error."
    
    return response

def fertilizer_planning_api(history, location=None, rainfall_forecast=None):
    """
    API function for fertilizer planning
    
    Args:
        history (list): List of fertilizer application history
        location (str, optional): Location for weather forecast
        rainfall_forecast (list, optional): 7-day rainfall forecast
        
    Returns:
        dict: Complete fertilizer recommendation
    """
    try:
        today_date = datetime.now()
        
        # Get the fertilizer recommendation
        recommendation = get_next_fertilizer_recommendation(
            history, today_date, location, rainfall_forecast
        )
        
        # Add today's date for reference
        recommendation["today"] = today_date.strftime("%Y-%m-%d")
        
        return recommendation
    
    except Exception as e:
        print(f"Error in fertilizer planning: {str(e)}")
        return {
            "error": str(e),
            "message": "An error occurred while generating fertilizer recommendations."
        }