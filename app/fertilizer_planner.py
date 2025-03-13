import datetime
from datetime import datetime, timedelta
from app.fertilizing_model import predict_7day_fertilizing_suitability, SINHALA_DAY_NAMES

# Define fertilizer names in both English and Sinhala
FERTILIZER_NAMES = {
    "Gliricidia leaves": "ග්ලිරිසීඩියා කොල",
    "Cow dung": "ගොම පොහොර",
    "NPK (10-10-10)": "NPK 10 අනුපාතයට",
    "Chicken manure": "කුකුල් පොහොර",
    "Compost": "කොම්පෝස්ට්"
}

# Reverse mapping for English to Sinhala
ENGLISH_TO_SINHALA = {
    english: sinhala for english, sinhala in FERTILIZER_NAMES.items()
}

# Sinhala to English mapping
SINHALA_TO_ENGLISH = {
    sinhala: english for english, sinhala in FERTILIZER_NAMES.items()
}

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
    },
    "Chicken manure": {
        "next": "Compost",
        "wait_days": 60  # 2 months
    },
    "Compost": {
        "next": "Gliricidia leaves",
        "wait_days": 90  # 3 months
    }
}

# Default fertilizer to start with if no history is available
DEFAULT_FERTILIZER = "Gliricidia leaves"

def get_sinhala_name(english_name):
    """Convert English fertilizer name to Sinhala"""
    return ENGLISH_TO_SINHALA.get(english_name, english_name)

def get_english_name(sinhala_name):
    """Convert Sinhala fertilizer name to English"""
    return SINHALA_TO_ENGLISH.get(sinhala_name, sinhala_name)

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
            
            # Check if fertilizer is in Sinhala and convert to English for processing
            english_fertilizer = get_english_name(fertilizer)
            
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
                    "fertilizer": english_fertilizer
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
        "next_fertilizer_sinhala": None,
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
        response["next_fertilizer_sinhala"] = get_sinhala_name(DEFAULT_FERTILIZER)
        response["recommended_date"] = effective_date.strftime("%Y-%m-%d")
        response["date_in_forecast"] = True
        
        if is_after_six_pm:
            response["message"] = "පොහොර යෙදීම හෙටින් ආරම්භ කරන්න"  # Start fertilizing from tomorrow
        else:
            response["message"] = "පොහොර යෙදීම අදින් ආරම්භ කරන්න"  # Start fertilizing from today
    else:
        # Get the most recent fertilizer application
        last_application = parsed_history[0]
        last_fertilizer_english = last_application["fertilizer"]
        last_fertilizer_sinhala = get_sinhala_name(last_fertilizer_english)
        
        response["last_application"] = {
            "date": last_application["date"].strftime("%Y-%m-%d"),
            "fertilizer": last_fertilizer_english,
            "fertilizer_sinhala": last_fertilizer_sinhala
        }
        
        # Find the next fertilizer in the rotation
        last_fertilizer = last_application["fertilizer"]
        
        # Handle case where the last fertilizer is not in our rotation chart
        if last_fertilizer not in FERTILIZER_ROTATION:
            next_fertilizer = DEFAULT_FERTILIZER
            wait_days = 30  # Default wait time for unknown fertilizers
            response["message"] = f"අඳුනා නොගත් අවසන් පොහොර වර්ගය: {last_fertilizer_sinhala}. දින 30 කින් {get_sinhala_name(next_fertilizer)} යොදන්න."
        else:
            # Get next fertilizer and wait period from rotation chart
            next_fertilizer = FERTILIZER_ROTATION[last_fertilizer]["next"]
            wait_days = FERTILIZER_ROTATION[last_fertilizer]["wait_days"]
        
        # Set the next fertilizer in both English and Sinhala
        response["next_fertilizer"] = next_fertilizer
        response["next_fertilizer_sinhala"] = get_sinhala_name(next_fertilizer)
        
        # Calculate the recommended date for the next application
        recommended_date = last_application["date"] + timedelta(days=wait_days)
        response["recommended_date"] = recommended_date.strftime("%Y-%m-%d")
        
        # Check if recommended date is today and it's after 6 PM
        if recommended_date.date() == today_date.date() and is_after_six_pm:
            # Move to tomorrow
            recommended_date = recommended_date + timedelta(days=1)
            response["recommended_date"] = recommended_date.strftime("%Y-%m-%d")
            response["message"] = f"හෙට {get_sinhala_name(next_fertilizer)} යෙදීම නිර්දේශ කරයි."
        # Check if recommended date has already passed
        elif recommended_date < effective_date:
            response["date_has_passed"] = True
            response["message"] = f"නිර්දේශිත දිනය ({response['recommended_date']}) දැනටමත් ගෙවී ගොස් ඇත. {get_sinhala_name(next_fertilizer)} ඉක්මනින් යොදන්න."
        else:
            days_until = (recommended_date - effective_date).days
            response["message"] = f"මීළඟ {get_sinhala_name(next_fertilizer)} යෙදීම දින {days_until} කින් නිර්දේශ කරයි."
    
    # If we have location and rainfall forecast, check weather suitability
    if location and rainfall_forecast and len(rainfall_forecast) > 0:
        # Get weather suitability for the next 7 days
        weather_forecast = predict_7day_fertilizing_suitability(location, rainfall_forecast)
        
        # Check if the weather forecast contains an error
        if len(weather_forecast) == 1 and 'error' in weather_forecast[0]:
            print(f"Weather forecast error: {weather_forecast[0]['error']}")
            response["message"] += " දෝෂයක් නිසා කාලගුණ යෝග්‍යතා තොරතුරු ලබා ගත නොහැක."
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
                                response["message"] += f" {recommended_date_str} දිනයේ කාලගුණය පොහොර යෙදීමට සුදුසුයි."
                            else:
                                response["message"] += f" {recommended_date_str} දිනයේ කාලගුණය පොහොර යෙදීමට සුදුසු නොවේ."
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
                            response["message"] += f" හොඳ කාලගුණයක් සහිත {best_day['date']} දිනයේ {get_sinhala_name(next_fertilizer)} යෙදීම නිර්දේශ කරයි."
                        else:
                            response["message"] += f" වඩා හොඳ කාලගුණික තත්වයන් සඳහා {best_day['date']} දක්වා කල් දැමීම සලකා බලන්න."
                    else:
                        response["message"] += " වත්මන් කාලගුණ අනාවැකියේ සුදුසු දින හමු නොවීය. වඩා හොඳ තත්වයන් සඳහා රැඳී සිටීමට සලකා බලන්න."
                
                # If recommended date is beyond the forecast period
                if not response["date_in_forecast"] and not response["date_has_passed"]:
                    response["message"] += f" නිර්දේශිත දිනය වත්මන් දින 7 කාලගුණ අනාවැකිය ඉක්මවා ඇත."
                    
            except Exception as e:
                print(f"Error processing weather forecast: {e}")
                response["message"] += " දෝෂයක් නිසා කාලගුණ යෝග්‍යතා තොරතුරු සැකසීමට නොහැකි විය."
    
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
            "message": "පොහොර නිර්දේශ උත්පාදනය කිරීමේදී දෝෂයක් ඇති විය."
        }