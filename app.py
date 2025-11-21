"""
Flask Web Application for Company RAG Chatbot
Features: Chat interface, PDF/Excel upload, Voice I/O
"""

from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
from pathlib import Path
from datetime import datetime
import logging
import tempfile

# RAG components
from src.data_loader import DocumentLoader
from src.embedding import EmbeddingManager
from src.vectorstore import VectorStore
from src.search import RAGRetriever, RAGPipeline
from voice import VoiceInterface

# Vertex AI LLM
import vertexai
from vertexai.generative_models import GenerativeModel
from dotenv import load_dotenv

# Configuration
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'temp-key')
app.config['UPLOAD_FOLDER'] = './data/pdf'
app.config['AUDIO_FOLDER'] = './data/audio'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'xlsx', 'xls'}
app.config['ALLOWED_AUDIO'] = {'wav', 'mp3', 'ogg', 'webm', 'm4a'}

# Create directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['AUDIO_FOLDER'], exist_ok=True)

# Global objects
rag_pipeline = None
embedding_manager = None
vector_store = None
doc_loader = None
voice_interface = None

# Conversation history storage (session-based)
conversation_histories = {}

PERSIST_DIR = os.path.abspath("data/vector_store")


def allowed_file(filename, allowed_set):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_set


def initialize_rag_system():
    """Initialize RAG system with Vertex AI"""
    global rag_pipeline, embedding_manager, vector_store, doc_loader, voice_interface
    os.makedirs(PERSIST_DIR, exist_ok=True)

    try:
        logger.info("🚀 Initializing RAG system...")

        # Initialize Vertex AI
        PROJECT_ID = os.getenv("PROJECT_ID")
        LOCATION = os.getenv("LOCATION", "us-central1")

        if not PROJECT_ID:
            logger.error("❌ PROJECT_ID not found in environment")
            return False

        vertexai.init(project=PROJECT_ID, location=LOCATION)
        llm = GenerativeModel("gemini-2.0-flash-exp")
        logger.info(f"✅ Initialized Vertex AI with project: {PROJECT_ID}")

        # Initialize components
        embedding_manager = EmbeddingManager(
            model_name="BAAI/bge-small-en-v1.5")
        vector_store = VectorStore(
            collection_name="company_documents",
            persist_directory=PERSIST_DIR
        )
        doc_loader = DocumentLoader(chunk_size=1000, chunk_overlap=200)

        # Initialize voice interface
        try:
            voice_interface = VoiceInterface()
            logger.info("✅ Voice interface initialized")
        except Exception as e:
            logger.warning(f"⚠️ Voice interface not available: {e}")
            voice_interface = None

        # Check collection
        count = vector_store.collection.count()
        logger.info(f"📚 Vector store contains {count} chunks")

        if count == 0:
            logger.warning("⚠️ Vector store is empty - upload PDFs first!")

        # Build retriever + pipeline
        retriever = RAGRetriever(vector_store, embedding_manager)
        rag_pipeline = RAGPipeline(
            retriever,
            llm,
            company_name=os.getenv("COMPANY_NAME", "Our Company")
        )

        logger.info("✅ RAG Pipeline initialized!")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to initialize: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def process_and_add_document(file_path: str):
    """Process uploaded document and add to vector store"""
    try:
        logger.info(f"📄 Processing: {file_path}")

        # Load document
        if file_path.endswith('.pdf'):
            documents = doc_loader.load_pdf(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            documents = doc_loader.load_excel(file_path)
        else:
            return False, "Unsupported file type"

        if not documents:
            return False, "No content extracted from file"

        # Split into chunks
        chunks = doc_loader.split_documents(documents)

        if not chunks:
            return False, "Document contains no readable text"

        # Generate embeddings
        texts = [doc.page_content for doc in chunks]
        embeddings = embedding_manager.generate_embeddings(texts)

        # Add to vector store
        vector_store.add_documents(chunks, embeddings)

        logger.info(f"✅ Added {len(chunks)} chunks")
        return True, f"Successfully added {len(chunks)} chunks"

    except Exception as e:
        logger.error(f"❌ Processing error: {e}")
        return False, str(e)


def get_conversation_history(session_id):
    """Get conversation history for a session"""
    if session_id not in conversation_histories:
        conversation_histories[session_id] = []
    return conversation_histories[session_id]


def add_to_conversation_history(session_id, user_msg, bot_response):
    """Add exchange to conversation history"""
    history = get_conversation_history(session_id)
    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": bot_response})
    # Keep only last 10 exchanges (20 messages)
    if len(history) > 20:
        conversation_histories[session_id] = history[-20:]


# ============================================
# ROUTES
# ============================================

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/api/check-admin', methods=['POST'])
def check_admin():
    """Validate superadmin password"""
    try:
        data = request.get_json()
        password = data.get('password', '').strip()
        correct_pass = os.getenv('SUPERADMIN_OTP')

        if not correct_pass:
            logger.error("SUPERADMIN_OTP not set in environment")
            return jsonify({'valid': False, 'error': 'Admin not configured'}), 500

        is_valid = password == correct_pass

        if is_valid:
            logger.info("✅ Admin authenticated successfully")
        else:
            logger.warning("❌ Failed admin authentication attempt")

        return jsonify({'valid': is_valid})

    except Exception as e:
        logger.error(f"Admin check error: {e}")
        return jsonify({'valid': False, 'error': str(e)}), 500


@app.route('/api/chat', methods=['POST'])
def chat():
    """Text-based chat endpoint with conversation memory"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        session_id = data.get('session_id', 'default')

        if not message:
            return jsonify({'error': 'Empty message'}), 400

        if not rag_pipeline:
            return jsonify({'error': 'RAG system not initialized'}), 500

        # Get conversation history
        history = get_conversation_history(session_id)

        # Build context from history
        context = ""
        if history:
            context = "Previous conversation:\n"
            for msg in history[-6:]:  # Last 3 exchanges
                role = "User" if msg["role"] == "user" else "Assistant"
                context += f"{role}: {msg['content']}\n"
            context += "\n"

        # Get response with context
        result = rag_pipeline.query(
            message,
            top_k=3,
            min_score=0.1,
            show_citations=data.get('show_citations', False),
        )

        # Save to history
        add_to_conversation_history(session_id, message, result['answer'])

        return jsonify({
            'answer': result['answer'],
            'sources': result['sources'],
            'confidence': result.get('confidence', 0.0),
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Chat error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/api/voice-query', methods=['POST'])
def voice_query():
    """Voice-based query endpoint (audio in, audio out)"""
    try:
        if not voice_interface:
            return jsonify({'error': 'Voice interface not available'}), 503

        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        audio_file = request.files['audio']
        voice = request.form.get("voice", "en-US-Neural2-J")  # <-- NEW

        logger.info(
            f"Received audio file: {audio_file.filename}, content_type: {audio_file.content_type}")
        logger.info(f"File size: {len(audio_file.read())} bytes")
        audio_file.seek(0)

        if audio_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Save uploaded audio
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        input_filename = f"input_{timestamp}.webm"
        input_path = os.path.join(app.config['AUDIO_FOLDER'], input_filename)
        audio_file.save(input_path)

        logger.info(f"🎤 Received audio: {input_path}")

        # Transcribe audio to text
        transcript = voice_interface.transcribe_audio(input_path)

        if not transcript:
            return jsonify({'error': 'Could not transcribe audio'}), 400

        logger.info(f"📝 Transcribed: {transcript}")

        # Get RAG response
        if not rag_pipeline:
            return jsonify({'error': 'RAG system not initialized'}), 500

        result = rag_pipeline.query(
            transcript,
            top_k=3,
            min_score=0.1,
            show_citations=False
        )

        answer_text = result['answer']
        logger.info(f"💬 Response: {answer_text[:100]}...")

        # Synthesize speech response
        output_filename = f"output_{timestamp}.mp3"
        output_path = os.path.join(app.config['AUDIO_FOLDER'], output_filename)

        success = voice_interface.synthesize_speech(
            answer_text,
            output_path,
            voice_name=voice
        )

        if not success:
            return jsonify({'error': 'Speech synthesis failed'}), 500

        logger.info(f"🔊 Audio generated: {output_path}")

        # Return audio file
        return send_file(
            output_path,
            mimetype='audio/mpeg',
            as_attachment=False,
            download_name=output_filename
        )

    except Exception as e:
        logger.error(f"Voice query error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload PDF/Excel document"""
    try:
        # Check superadmin password
        superadmin_pass = request.form.get('superadmin_password')
        if not superadmin_pass:
            superadmin_pass = request.headers.get('X-Admin-Key')

        correct_pass = os.getenv('SUPERADMIN_OTP')

        if correct_pass and (superadmin_pass is None or superadmin_pass != correct_pass):
            logger.warning("❌ Unauthorized upload attempt")
            return jsonify({'error': 'Unauthorized'}), 401

        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename, app.config['ALLOWED_EXTENSIONS']):
            return jsonify({'error': 'Invalid file type. Only PDF and Excel allowed'}), 400

        # Save file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{secure_filename(file.filename)}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        logger.info(f"📁 Saved: {filepath}")

        # Process and add to vector store
        success, message = process_and_add_document(filepath)

        if success:
            return jsonify({
                'message': 'File uploaded and processed successfully',
                'filename': filename,
                'details': message
            })
        else:
            return jsonify({'error': f'Processing failed: {message}'}), 500

    except Exception as e:
        logger.error(f"Upload error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/api/documents', methods=['GET'])
def list_documents():
    """List all uploaded documents"""
    try:
        upload_dir = Path(app.config['UPLOAD_FOLDER'])
        files = []
        for file_path in upload_dir.glob('*'):
            if file_path.is_file():
                files.append({
                    'name': file_path.name,
                    'size': file_path.stat().st_size,
                    'uploaded': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                })
        return jsonify({'documents': files})
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats')
def stats():
    """Get system statistics"""
    try:
        if not vector_store:
            return jsonify({'error': 'Vector store not initialized'}), 500

        return jsonify({
            'total_documents': vector_store.collection.count(),
            'collection_name': vector_store.collection_name,
            'status': 'ready',
            'voice_enabled': voice_interface is not None
        })

    except Exception as e:
        logger.error(f"Stats error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'rag_initialized': rag_pipeline is not None,
        'voice_available': voice_interface is not None
    })


@app.before_request
def before_first_request():
    """Initialize RAG system before first request"""
    global rag_pipeline
    if rag_pipeline is None:
        initialize_rag_system()


# ============================================
# STARTUP
# ============================================

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5001))
    if initialize_rag_system():
        logger.info(f"🌐 Starting Flask server on port {port}...")
        app.run(host='0.0.0.0', port=port, debug=True)
    else:
        logger.error("❌ Failed to initialize. Exiting.")
