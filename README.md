# Call Center Voice AI

This project is an AI-powered voice assistant for call centers, capable of transcribing user queries, retrieving relevant responses using FAISS, and generating improved replies using OpenAI's GPT model. It also includes text-to-speech (TTS) functionality for spoken responses.

## Features
- 🎤 **Speech Recognition:** Converts user speech to text using Google Speech Recognition.
- 🔍 **Semantic Search:** Uses FAISS for fast retrieval of the most relevant predefined response.
- 🤖 **AI-Powered Response:** Enhances responses using OpenAI's GPT model.
- 🔊 **Text-to-Speech (TTS):** Speaks responses using `pyttsx3`.
- 🏎 **Multi-threading:** Ensures real-time interaction.

## Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/alihassanml/Call-center-voice-ai.git
cd Call-center-voice-ai
```

### 2️⃣ Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Set Up Environment Variables
Create a `.env` file and add your OpenAI API key:
```env
OPEN_AI=your_openai_api_key
```

## Usage
Run the AI assistant:
```bash
python main.py
```
The assistant will start listening for voice input and provide intelligent responses.

## File Structure
```
📂 Call-center-voice-ai
├── 📜 main.py               # Main script
├── 📜 requirements.txt      # Dependencies
├── 📜 .env                  # Environment variables (ignored in Git)
├── 📜 data.csv              # Dataset of predefined responses
├── 📜 vectorizer.pkl        # TF-IDF vectorizer model
├── 📜 call_center_faiss.index  # FAISS index for fast response retrieval
└── 📜 README.md             # Project documentation
```

## Technologies Used
- **Python 3.11**
- **FAISS** (Fast Approximate Nearest Neighbors)
- **OpenAI GPT-3.5 Turbo**
- **SpeechRecognition** (Google Speech-to-Text)
- **pyttsx3** (Text-to-Speech)
- **Scikit-learn** (TF-IDF Vectorization)
- **Threading** (Concurrent execution)

## Contributing
Feel free to submit pull requests or report issues in the [GitHub repository](https://github.com/alihassanml/Call-center-voice-ai).

## License
This project is open-source and available under the [MIT License](LICENSE).