# streamlit_app.py

import streamlit as st
import nltk
import numpy as np
import json
import pickle
from nltk.stem.lancaster import LancasterStemmer
from tensorflow.keras.models import load_model
import re 
import numpy.random as np_random # Use numpy.random for random choices

# -----------------------------------------------------
# CRITICAL FIX: DOWNLOAD NLTK DATA FOR STREAMLIT CLOUD
# -----------------------------------------------------
# This ensures the 'punkt' tokenizer data is available on the cloud server.
try:
    # Check if 'punkt' data is already available
    nltk.data.find('tokenizers/punkt')
except LookupError: # <--- CHANGED TO CATCH LookupError which is raised by nltk.data.find()
    # If not found, download it. This only runs the first time the app starts.
    nltk.download('punkt')
# -----------------------------------------------------
# END NLTK FIX
# -----------------------------------------------------


# Initialize Stemmer
stemmer = LancasterStemmer()

# --- INITIAL SETUP (Load files and core functions) ---

@st.cache_resource
def load_data():
    """Load the model and all data files once when the app starts."""
    try:
        data = pickle.load(open("training_data.pkl", "rb"))
        words = data['words']
        classes = data['classes']
        model = load_model('chatbot_model.keras')
        
        with open('intents.json', encoding='utf-8') as file:
            intents_data = json.load(file)
            
        with open('college_data.json', encoding='utf-8') as file:
            college_data = json.load(file)
            
        return words, classes, model, intents_data, college_data
    except Exception as e:
        # If loading fails, raise a Streamlit error to stop the app
        st.error(f"Failed to load required data files. Please ensure all files (model.keras, .pkl, .json) are present. Error: {e}")
        st.stop()


# Load all resources
words, classes, model, intents_data, college_data = load_data()


# --- CORE NLP AND FACTUAL FUNCTIONS ---

def clean_up_sentence(sentence):
    """Tokenize and stem the sentence."""
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words):
    """Convert sentence to bag of words array."""
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)  
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s: 
                bag[i] = 1
    return(np.array(bag))

def classify_local(sentence):
    """Classify the intent of the sentence."""
    p = bow(sentence, words)
    # The model expects a batch of inputs, so we wrap p in an array
    results = model.predict(np.array([p]), verbose=0)[0] 
    # Filter out predictions below a threshold (e.g., 0.7)
    results = [[i, r] for i, r in enumerate(results) if r > 0.7] 
    
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

# --- UPDATED FUNCTION TO RETRIEVE FACTS (Added user_message parameter) ---
def get_factual_response(tag, data, user_message=""):
    """Retrieves detailed facts based on the predicted tag."""
    
    # Access key data structures
    accreditation = data.get('accreditation_and_rankings', {})
    admin_support = data.get('administrative_and_student_support', {})
    admin_personnel = admin_support.get('key_personnel', {})
    admin_deans = admin_support.get('deans_and_coordinators', {})
    infra = data.get('infrastructure_and_facilities', {}).get('campus_details', {})
    hostel = data.get('infrastructure_and_facilities', {}).get('hostel_facilities', {})
    programs = data.get('academic_programs_summary', {})
    placements = data.get('placements_and_alumni', {})
    
    
    if tag == "college_name_query":
        return f"The full name is **{data.get('college_name', 'Nehru Memorial College')} (Autonomous)**. It's located at {data.get('location', 'Puthanampatti')}."

    # Logic for Principal, Secretary, and VP (Now uses user_message for specificity)
    elif tag == "principal_query":
        principal = admin_personnel.get('principal', 'N/A') 
        president = admin_personnel.get('president', 'N/A')
        secretary = admin_personnel.get('secretary', 'N/A')
        vice_principal = admin_personnel.get('vice_principal', 'N/A')
        
        lower_msg = user_message.lower()
        
        # Smart lookup based on keywords in the original message
        if "principal" in lower_msg and "vice" not in lower_msg:
            return f"The **Principal** of NMC is **{principal}**."
        elif "secretary" in lower_msg:
            return f"The **Secretary** of NMC is **{secretary}**."
        elif "president" in lower_msg:
            return f"The **President** of NMC is **{president}**."
        elif "vice principal" in lower_msg or "vp" in lower_msg:
            return f"The **Vice Principal** of NMC is **{vice_principal}**."
        
        # Default response if no specific keyword is found
        return (
            f"The key administrative personnel are:\n"
            f"- **Principal:** {principal}\n"
            f"- **President:** {president}\n"
            f"- **Secretary:** {secretary}\n"
            f"- **Vice Principal:** {vice_principal}"
        )

    elif tag == "facilities_query":
        unique_facilities = ', '.join(data.get('infrastructure_and_facilities', {}).get('unique_facilities', [])[:3])
        return (
            f"NMC campus has a built-up area of **{infra.get('built_up_area', 'N/A')} Sqmt**, {infra.get('classrooms', 'N/A')} classrooms, and {infra.get('computer_centre_pcs', 'N/A')} PCs.\n"
            f"Unique facilities include: {unique_facilities}."
        )

    elif tag == "hostel_query":
        hostel_boys = hostel.get('boys_hostel', {})
        hostel_girls = hostel.get('girls_hostel', {})
        
        return (
            f"Hostel Details:\n"
            f"- **Boys Hostel ({hostel_boys.get('name', 'N/A')}):** Capacity of {hostel_boys.get('capacity', 'N/A')}.\n"
            f"- **Girls Hostel ({hostel_girls.get('name', 'N/A')}):** Capacity of {hostel_girls.get('capacity', 'N/A')}."
        )

    elif tag == "programs_count_query":
        counts = programs.get('total_programs_count', {})
        ug = counts.get('ug', 'N/A')
        pg = counts.get('pg', 'N/A')
        
        return (
            f"NMC offers approximately **{programs.get('total_courses_offered', 'N/A')}**.\n"
            f"Currently, there are **{ug} UG** and **{pg} PG** programs."
        )
        
    elif tag == "pg_query":
        pg_list = programs.get('pg_courses', [])
        pg_display = ', '.join(pg_list[:5]) + '...' if pg_list else 'N/A'
        return f"PG courses include: **{pg_display}**. The college also offers MCA and MBA (AICTE Approved)."

    # Logic for Data Science
    elif tag == "data_science_query":
        ds_dept = next((dept for dept in data.get('departments_details', []) if dept['department_name'] == "Data Science"), None)
        hod = data.get('key_heads_of_department_full_list', {}).get('HOD_Data_Science_SF', 'N/A')
        
        if ds_dept:
            return (
                f"Data Science is a Self-Financed program. NMC was a **Pioneer** in introducing it (2017-2018).\n"
                f"Programs: {', '.join(ds_dept.get('ug_programs', []))} and {', '.join(ds_dept.get('pg_programs', []))}.\n"
                f"HOD is: **{hod}**."
            )
        return "Specific Data Science department details are currently unavailable in the knowledge base."


    elif tag == "placement_query":
        stats = placements.get('placement_stats_2023_24', {})
        return (
            f"Placement highlights (2023-24):\n"
            f"- **UG Placed:** {stats.get('ug_placed_percentage', 'N/A')}\n"
            f"- **Highest Salary:** {stats.get('highest_salary', 'N/A')}\n"
            f"- **Total Placements:** {stats.get('total_domestic_placements', 'N/A')}"
        )
        
    elif tag == "accreditation_query":
        return f"The college holds an **'{accreditation.get('naac_accreditation', 'N/A')}'** by NAAC in the Second Cycle (2022)."
        
    # Logic for All Departments
    elif tag == "all_departments_query":
        departments_list = data.get('departments_details', [])
        response = "NMC offers a wide range of programs. Key departments include:\n"
        
        for dept in departments_list:
            name = dept.get('department_name', 'N/A')
            ug = ', '.join(dept.get('ug_programs', [])) if dept.get('ug_programs') else 'N/A'
            pg = ', '.join(dept.get('pg_programs', [])) if dept.get('pg_programs') else 'N/A'
            
            if name == "Other Allied/SF Programs": continue 

            response += f"\n**{name}**:\n  UG: {ug}\n  PG: {pg}"
            
        return response.strip()

    # Logic for All HOD Names
    elif tag == "all_hods_query":
        hod_dict = data.get('key_heads_of_department_full_list', {})
        
        if not hod_dict:
            return "I am unable to retrieve the list of HODs at this moment."
            
        response = "**Heads of Departments (HODs):**\n"
        
        for key, name in hod_dict.items():
            # Clean up the key name for display (e.g., HOD_Physics_Aided -> Physics)
            display_name = re.sub(r'HOD_|DIRECTOR_|_Aided|_SF', '', key).replace('_', ' ')
            response += f"- **{display_name.title()}:** {name}\n"
            
        return response.strip()
        
    # Logic for All Key Administrative Staff
    elif tag == "all_admin_query":
        response = "**Key Administrative Personnel:**\n"
        
        # 1. Add top-level personnel
        for title, name in admin_personnel.items():
            response += f"- **{title.replace('_', ' ').title()}:** {name}\n"
            
        response += "\n**Deans and Coordinators:**\n"
        
        # 2. Add Deans and Coordinators
        for title, name in admin_deans.items():
            response += f"- **{title.replace('_', ' ').title()}:** {name}\n"
            
        return response.strip()

    return None # Return None if tag is not a factual query


def get_response(ints, intents_json, college_data, user_message):
    """Retrieve a random response based on the classified intent, handling facts separately."""
    if not ints:
        return "I didn't quite understand that. Can you rephrase or ask about the college general information?"
    
    tag = ints[0]['intent']
    
    # 1. Check if the intent is a Factual Query
    if tag.endswith("_query"):
        # Pass the user message to allow for specific lookups (e.g., "who is the secretary?")
        factual_answer = get_factual_response(tag, college_data, user_message)
        if factual_answer:
            return factual_answer
    
    # 2. Handle standard conversational intents
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = np_random.choice(i['responses'])
            
            if result.startswith("GET_"):
                # This should not happen with the current setup
                return "I recognized the intent, but the dynamic lookup failed (Internal logic error)."
            
            return result
            
    return "Sorry, I can't find a response for that specific topic."


# --- STREAMLIT CHAT INTERFACE ---

st.set_page_config(page_title="NMC Chatbot", layout="wide")
st.title("🏛️ NMC College Chatbot")
st.markdown("Ask me about facilities, courses, HODs, or placements!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add initial bot message
    st.session_state.messages.append({"role": "assistant", 
                                      "content": "Hello! I'm the NMC Chatbot. Ask me about general college information, like **'facilities'**, **'principal'**, or **'courses'**."})

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question about NMC..."):
    # 1. Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Get bot response
    with st.spinner("Thinking..."):
        ints = classify_local(prompt)
        # --- CRITICAL: PASS 'prompt' HERE for specific intent handling ---
        res = get_response(ints, intents_data, college_data, prompt) 
    
    # 3. Display assistant response
    with st.chat_message("assistant"):
        st.session_state.messages.append({"role": "assistant", "content": res})
        st.markdown(res)