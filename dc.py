import argparse
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Conv1D, Dropout, Input, Concatenate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import numpy as np
import os
import json
import collections # For deque
import subprocess # Added for gs
import tempfile # Added for temporary file creation

# --- Global Configuration / Constants ---
DEFAULT_MAX_VOCAB_SIZE = 102400 
DEFAULT_EMBEDDING_DIM = 32
DEFAULT_DROPOUT_RATE = 0.6
DEFAULT_EPOCHS = 5
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_WINDOW_SIZE = 1024 # Default window size for CNN input
DEFAULT_STRIDE = 256    # Default stride for overlapping windows
DEFAULT_CNN_FILTERS = 128 # Default number of filters for Conv1D layer
DEFAULT_CNN_KERNEL_SIZE = 5 # Default kernel size for Conv1D layer

# --- Helper Functions ---

def preprocess_text(text):
    """Simple text cleaning: lowercase, remove non-alphanumeric (except spaces)."""
    text = text.lower()
    text = ''.join([char for char in text if char.isalnum() or char.isspace()])
    return text

def get_document_text(file_path):
    """
    Reads and extracts text content from a given document file.
    Currently supports plain text files (.txt) and PDF files (.pdf) using Ghostscript (gs).
    Args:
        file_path (str): The full path to the document file.
    Returns:
        str: The extracted text content.
    Raises:
        ValueError: If the file type is not supported.
        FileNotFoundError: If the file does not exist.
        IOError: For other file reading errors, including Ghostscript execution issues.
    """
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == '.txt':
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    elif file_extension == '.pdf':
        temp_txt_file = None
        try:
            # Create a temporary file to store the extracted text
            # Ghostscript often writes to a file rather than stdout for text extraction
            with tempfile.NamedTemporaryFile(mode='w+', delete=False, encoding='utf-8', suffix='.txt') as tmp_f:
                temp_txt_file = tmp_f.name

            # Ghostscript command for text extraction (using txtwrite device)
            # gs -sDEVICE=txtwrite -o output.txt input.pdf
            command = [
                'gs',
                '-sDEVICE=txtwrite',
                '-o', temp_txt_file,
                file_path
            ]
            
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True, # Raises CalledProcessError for non-zero exit codes
                encoding='utf-8'
            )
            
            # Read the extracted text from the temporary file
            with open(temp_txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                extracted_text = f.read()
            return extracted_text

        except FileNotFoundError:
            raise IOError("Ghostscript ('gs') is not installed or not found in PATH. Please install Ghostscript (https://www.ghostscript.com/download/) to process PDFs.")
        except subprocess.CalledProcessError as e:
            # This might happen if the PDF is corrupted or gs has issues
            raise IOError(f"Error processing PDF {file_path} with Ghostscript: {e}\nStdout: {e.stdout}\nStderr: {e.stderr}")
        except Exception as e:
            raise IOError(f"An unexpected error occurred while processing PDF {file_path} with Ghostscript: {e}")
        finally:
            # Clean up the temporary file
            if temp_txt_file and os.path.exists(temp_txt_file):
                os.remove(temp_txt_file)
    # elif file_extension == '.docx':
    #     # Example (requires python-docx library):
    #     try:
    #         import docx
    #         doc = docx.Document(file_path)
    #         return "\n".join([para.text for para in doc.paragraphs])
    #     except ImportError:
    #         raise ImportError("python-docx is not installed. Please install it to process DOCX files.")
    #     except Exception as e:
    #         raise IOError(f"Error reading DOCX {file_path}: {e}")
    else:
        raise ValueError(f"Unsupported file type for {file_path}: {file_extension}")


def _get_windows_from_text(tokenizer, text, window_size, stride):
    """
    Tokenizes text and breaks it into overlapping windows.
    Args:
        tokenizer (tf.keras.preprocessing.text.Tokenizer): The tokenizer to use.
        text (str): The input text.
        window_size (int): The size of each window (sequence length).
        stride (int): The step size for moving the window.
    Returns:
        list: A list of padded tokenized sequences (windows).
    """
    processed_text = preprocess_text(text)
    tokenized_sequence = tokenizer.texts_to_sequences([processed_text])[0] # Get the list of token IDs

    windows = []
    if not tokenized_sequence:
        # Handle empty or untokenizable text
        # Return one empty window to maintain structure, will be padded to window_size
        windows.append(pad_sequences([[]], maxlen=window_size, padding='post', truncating='post')[0])
        return windows

    # Generate windows
    for i in range(0, len(tokenized_sequence), stride):
        window = tokenized_sequence[i:i + window_size]
        windows.append(window)

    # Pad all windows to the specified window_size
    padded_windows = pad_sequences(windows, maxlen=window_size, padding='post', truncating='post')
    return padded_windows.tolist() # Convert to list of lists for easier handling


def build_cnn_model(vocab_size, embedding_dim, window_size, num_classes, dropout_rate=DEFAULT_DROPOUT_RATE, filters=DEFAULT_CNN_FILTERS, kernel_size=DEFAULT_CNN_KERNEL_SIZE):
    """
    Builds a simple Convolutional Neural Network for text classification.
    Args:
        vocab_size (int): Size of the vocabulary.
        embedding_dim (int): Dimension of the word embeddings.
        window_size (int): The fixed input length for each window.
        num_classes (int): Number of output classes for the softmax layer.
        dropout_rate (float): Dropout rate for regularization.
        filters (int): Number of filters for the Conv1D layer.
        kernel_size (int): Kernel size for the Conv1D layer.
    Returns:
        tf.keras.Model: Compiled CNN model.
    """
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=window_size),
        Conv1D(filters, kernel_size, activation='relu'),
        GlobalAveragePooling1D(), # Reduces dimension, good for classification
        Dense(128, activation='relu'),
        Dropout(dropout_rate),
        Dense(num_classes, activation='softmax')
    ])
    return model

def build_decision_network(input_dim, num_output_classes, dropout_rate=DEFAULT_DROPOUT_RATE):
    """
    Builds the two-layer fully-connected decision network with (sqrt(n+1)+1) nodes
    and skip connections from input and first hidden layer to the output layer.
    Args:
        input_dim (int): Total number of features (concatenated outputs from parent CNN and child CNNs).
        num_output_classes (int): Number of decision outcomes (e.g., pass_to_SubA, retain_at_CurrentLevel).
        dropout_rate (float): Dropout rate for regularization.
    Returns:
        tf.keras.Model: Compiled Decision Network model.
    """
    # Calculate nodes for hidden layers based on sqrt(n+1)+1
    # Ensure at least 2 nodes for very small input_dim
    hidden_nodes = int(np.sqrt(input_dim + 1) + 1)
    if hidden_nodes < 2:
        hidden_nodes = 2

    # Use Keras Functional API for skip connections
    input_layer = Input(shape=(input_dim,))

    # First hidden layer
    hidden_layer_1 = Dense(hidden_nodes, activation='relu')(input_layer)
    hidden_layer_1_dropout = Dropout(dropout_rate)(hidden_layer_1)

    # Second hidden layer
    hidden_layer_2 = Dense(hidden_nodes, activation='relu')(hidden_layer_1_dropout)

    # Concatenate input, first hidden layer output, and second hidden layer output
    # This creates the "weighted skip-connection" to the final layer
    concatenated_features = Concatenate()([input_layer, hidden_layer_1, hidden_layer_2])

    # Output layer
    output_layer = Dense(num_output_classes, activation='softmax')(concatenated_features)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model

def _collect_all_documents_with_paths(root_dir):
    """
    Recursively collects all document texts and their full hierarchical path segments.
    Skips directories that start with an underscore.
    Args:
        root_dir (str): The root directory from which to start collecting documents.
    Returns:
        dict: A dictionary where keys are full file paths and values are dictionaries
              containing 'text' (document content) and 'path_segments' (list of
              category names in the hierarchical path, including the filename).
    """
    all_documents_data = {}
    print(f"Collecting documents from root directory: {root_dir}")
    for root, dirs, files in os.walk(root_dir):
        print(f"  Processing directory: {root}")
        # Filter out directories starting with an underscore
        # Modifying dirs in-place tells os.walk not to recurse into them
        original_dirs = list(dirs) # Make a copy to iterate
        dirs[:] = [d for d in dirs if not d.startswith('_')]
        skipped_dirs = [d for d in original_dirs if d.startswith('_')]
        if skipped_dirs:
            print(f"    Skipping subdirectories: {', '.join(skipped_dirs)}")

        for file_name in files:
    
            full_file_path = os.path.join(root, file_name)
            try:
                # Use the new get_document_text function
                doc_content = get_document_text(full_file_path)
                
                # Determine path segments relative to the root_dir
                relative_path = os.path.relpath(full_file_path, root_dir)
                path_segments = relative_path.split(os.sep)
                all_documents_data[full_file_path] = {
                    'text': doc_content,
                    'path_segments': path_segments
                }
                print(f"    Collected document: {full_file_path}")
            except (ValueError, FileNotFoundError, IOError) as e:
                print(f"    Warning: Could not read {full_file_path}: {e}")
    print(f"Finished collecting documents. Total documents: {len(all_documents_data)}")
    return all_documents_data

# --- Phase 1: Train Folder CNNs ---
def _train_folder_cnn_for_node(current_data_node_path, model_output_path, all_documents_data, args):
    """
    Trains a CNN for a specific node to classify documents into its immediate children
    (subfolders or direct documents). Documents are processed in windows with sample weighting.
    Args:
        current_data_node_path (str): The path to the current data directory (node).
        model_output_path (str): The path where the trained CNN model and its assets will be saved.
        all_documents_data (dict): A dictionary of all documents and their full paths.
        args (argparse.Namespace): Command-line arguments containing hyperparameters.
    Returns:
        tuple: (tf.keras.Model, tf.keras.preprocessing.text.Tokenizer, sklearn.preprocessing.LabelEncoder)
               or (None, None, None) if training is skipped.
    """
    print(f"  Training Folder CNN for: {current_data_node_path}")

    documents_for_cnn_raw_text = []
    documents_for_cnn_original_paths = [] # To link back to true labels for each document
    
    # Collect all documents that fall under this node's domain
    for full_doc_path, doc_data in all_documents_data.items():
        if full_doc_path.startswith(current_data_node_path):
            documents_for_cnn_raw_text.append(doc_data['text'])
            documents_for_cnn_original_paths.append(full_doc_path)

    if not documents_for_cnn_raw_text:
        print(f"  Skipping Folder CNN for {current_data_node_path}: No documents found for this node.")
        return None, None, None

    # Initialize tokenizer and fit on all relevant documents for this CNN
    tokenizer = Tokenizer(num_words=args.max_vocab_size, oov_token="<unk>")
    tokenizer.fit_on_texts([preprocess_text(doc) for doc in documents_for_cnn_raw_text])
    vocab_size = len(tokenizer.word_index) + 1 # +1 for OOV token

    # Prepare windows and labels for training
    all_windows = []
    all_window_labels = []
    all_sample_weights = []

    for i, doc_text in enumerate(documents_for_cnn_raw_text):
        full_doc_path = documents_for_cnn_original_paths[i]
        
        # Determine the immediate child name (label) for this document
        relative_to_node = os.path.relpath(full_doc_path, current_data_node_path)
        if os.sep in relative_to_node:
            immediate_child_name = relative_to_node.split(os.sep)[0]
        else:
            immediate_child_name = relative_to_node # The filename itself

        # Skip documents that are in underscore-prefixed folders
        if immediate_child_name.startswith('_'):
            print(f"    Skipping document {os.path.basename(full_doc_path)}: In underscore-prefixed folder.")
            continue

        doc_windows = _get_windows_from_text(tokenizer, doc_text, args.window_size, args.stride)
        
        if not doc_windows:
            print(f"    Warning: Document {os.path.basename(full_doc_path)} produced no valid windows. Skipping.")
            continue

        num_windows_in_doc = len(doc_windows)
        weight_per_window = 1.0 / num_windows_in_doc # Each document contributes 1 to the loss

        all_windows.extend(doc_windows)
        all_window_labels.extend([immediate_child_name] * num_windows_in_doc)
        all_sample_weights.extend([weight_per_window] * num_windows_in_doc)

    if not all_windows or len(set(all_window_labels)) < 2:
        print(f"  Skipping Folder CNN for {current_data_node_path}: Insufficient data or classes after windowing ({len(set(all_window_labels))}).")
        return None, None, None 

    label_encoder = LabelEncoder()
    integer_labels = label_encoder.fit_transform(all_window_labels)
    num_classes = len(label_encoder.classes_)
    one_hot_labels = tf.keras.utils.to_categorical(integer_labels, num_classes=num_classes)

    padded_windows_array = np.array(all_windows)
    sample_weights_array = np.array(all_sample_weights)

    # Build and train CNN
    model = build_cnn_model(vocab_size, args.embedding_dim, args.window_size, num_classes, 
                            args.dropout_rate, args.cnn_filters, args.cnn_kernel_size) # Pass new args
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    X_train, X_val, y_train, y_val, sample_weights_train, sample_weights_val = train_test_split(
        padded_windows_array, one_hot_labels, sample_weights_array, test_size=0.2, random_state=42
    )
    
    print(f"    Training Folder CNN for {current_data_node_path} with {len(X_train)} training windows ({len(set(documents_for_cnn_original_paths))} documents) and {len(X_val)} validation windows...")
    print(f"    Model Summary for Folder CNN at {current_data_node_path}:")
    model.summary(print_fn=print) # Print model summary
    
    model.fit(X_train, y_train, sample_weight=sample_weights_train, # Pass sample weights here
              epochs=args.epochs, batch_size=args.batch_size, 
              validation_data=(X_val, y_val, sample_weights_val), # Pass sample weights for validation
              verbose=1)
    
    loss, accuracy = model.evaluate(X_val, y_val, sample_weight=sample_weights_val, verbose=0) # Evaluate with sample weights
    print(f"    Folder CNN Validation Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")


    # Save model and assets
    os.makedirs(model_output_path, exist_ok=True)
    model.save(os.path.join(model_output_path, 'model.keras'))
    with open(os.path.join(model_output_path, 'tokenizer_config.json'), 'w') as f:
        json.dump(tokenizer.to_json(), f)
    with open(os.path.join(model_output_path, 'label_encoder_classes.json'), 'w') as f:
        json.dump(list(label_encoder.classes_), f)
    
    # Save CNN specific metadata (vocab_size, window_size, stride, filters, kernel_size)
    cnn_metadata = {
        'vocab_size': vocab_size,
        'window_size': args.window_size,
        'stride': args.stride,
        'filters': args.cnn_filters, # Added filters
        'kernel_size': args.cnn_kernel_size # Added kernel_size
    }
    with open(os.path.join(model_output_path, 'cnn_metadata.json'), 'w') as f:
        json.dump(cnn_metadata, f)

    print(f"  Folder CNN saved to: {model_output_path}")
    return model, tokenizer, label_encoder

# --- Phase 2: Train Decision Networks ---

def _prepare_decision_network_data(current_data_node_path, current_model_node_path, root_data_dir, model_base_dir, all_documents_data, args):
    """
    Prepares training data for the Decision Network at current_model_node_path.
    This involves running documents through the current node's CNN and relevant child CNNs,
    aggregating window-level predictions to document-level features.
    Args:
        current_data_node_path (str): Path to the current data directory.
        current_model_node_path (str): Path to the current model directory.
        root_data_dir (str): The absolute root of the original document hierarchy.
        model_base_dir (str): The absolute base directory where all models are stored.
        all_documents_data (dict): Dictionary of all documents and their paths.
        args (argparse.Namespace): Command-line arguments.
    Returns:
        tuple: (numpy.ndarray, numpy.ndarray, sklearn.preprocessing.LabelEncoder, dict)
               or (None, None, None, None) if preparation fails or no data.
    """
    print(f"  Preparing Decision Network data for: {current_data_node_path}")

    dn_inputs = []
    dn_labels = []
    documents_processed_for_dn = 0

    # Load current node's Folder CNN and tokenizer
    try:
        folder_cnn_model = load_model(os.path.join(current_model_node_path, 'model.keras'))
        with open(os.path.join(current_model_node_path, 'tokenizer_config.json'), 'r') as f:
            folder_cnn_tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(json.load(f))
        with open(os.path.join(current_model_node_path, 'label_encoder_classes.json'), 'r') as f:
            folder_cnn_label_classes = json.load(f)
            folder_cnn_label_encoder = LabelEncoder()
            folder_cnn_label_encoder.classes_ = np.array(folder_cnn_label_classes)
        with open(os.path.join(current_model_node_path, 'cnn_metadata.json'), 'r') as f:
            folder_cnn_metadata = json.load(f)
        current_cnn_window_size = folder_cnn_metadata['window_size']
        current_cnn_stride = folder_cnn_metadata['stride']
        # Load filters and kernel_size for consistency (though not directly used here)
        current_cnn_filters = folder_cnn_metadata.get('filters', DEFAULT_CNN_FILTERS)
        current_cnn_kernel_size = folder_cnn_metadata.get('kernel_size', DEFAULT_CNN_KERNEL_SIZE)

    except Exception as e:
        print(f"  Error loading Folder CNN for {current_data_node_path} to prepare DN data: {e}")
        return None, None, None, None, None

    # Determine the number of classes for the current Folder CNN (output dimension)
    current_cnn_output_dim = len(folder_cnn_label_encoder.classes_)

    # Load child Folder CNNs and their tokenizers/metadata
    child_cnn_models = {}
    child_cnn_tokenizers = {}
    child_cnn_metadata_loaded = {} # To store window_size, stride, filters, kernel_size for each child CNN

    # Get immediate subfolders in the *data* directory to know which children to expect
    # Filter out subfolders starting with an underscore
    immediate_subfolders = [d for d in os.listdir(current_data_node_path) if os.path.isdir(os.path.join(current_data_node_path, d)) and not d.startswith('_')]

    for subfolder_name in immediate_subfolders:
        child_model_dir = os.path.join(current_model_node_path, subfolder_name)
        child_model_path = os.path.join(child_model_dir, 'model.keras')
        child_tokenizer_path = os.path.join(child_model_dir, 'tokenizer_config.json')
        child_metadata_path = os.path.join(child_model_dir, 'cnn_metadata.json')
        
        if os.path.exists(child_model_path) and os.path.exists(child_tokenizer_path) and os.path.exists(child_metadata_path):
            try:
                child_cnn_model = load_model(child_model_path)
                with open(child_tokenizer_path, 'r') as f:
                    child_cnn_tokenizers[subfolder_name] = tf.keras.preprocessing.text.tokenizer_from_json(json.load(f))
                with open(child_metadata_path, 'r') as f:
                    child_cnn_metadata_loaded[subfolder_name] = json.load(f)
                
                child_cnn_models[subfolder_name] = child_cnn_model
                print(f"    Loaded child CNN for {subfolder_name}")
            except Exception as e:
                print(f"    Warning: Could not load child CNN for {subfolder_name} at {current_model_node_path}: {e}")
                child_cnn_metadata_loaded[subfolder_name] = {'window_size': 0, 'stride': 0, 'vocab_size': 0, 'filters': 0, 'kernel_size': 0} # Mark as invalid
        else:
            child_cnn_metadata_loaded[subfolder_name] = {'window_size': 0, 'stride': 0, 'vocab_size': 0, 'filters': 0, 'kernel_size': 0} # Mark as invalid
            print(f"    Child CNN model files not found for {subfolder_name}. Assuming 0 output dimension for DN input.")


    # Calculate the total input dimension for the Decision Network
    # This sum includes the current CNN's output + all *expected* child CNN outputs
    total_dn_input_dim = current_cnn_output_dim
    # We need to store the *expected* output dimensions of children for the DN metadata,
    # even if the child model failed to load during DN prep, to ensure consistent DN input shape.
    expected_child_cnn_output_dims_for_metadata = {}
    for subfolder_name in immediate_subfolders:
        if subfolder_name in child_cnn_models:
            child_output_dim = child_cnn_models[subfolder_name].layers[-1].output_shape[-1]
            total_dn_input_dim += child_output_dim
            expected_child_cnn_output_dims_for_metadata[subfolder_name] = child_output_dim
        else:
            # If child model not loaded, its contribution to the DN input is 0 for this run,
            # but we record 0 as its expected output dim for the DN's metadata consistency.
            expected_child_cnn_output_dims_for_metadata[subfolder_name] = 0 


    print(f"    Calculated Decision Network input dimension: {total_dn_input_dim}")

    # Iterate through all documents relevant to this node to generate DN training samples
    for full_doc_path, doc_data in all_documents_data.items():
        # Check if this document belongs to the current node's domain
        if not full_doc_path.startswith(current_data_node_path):
            continue

        doc_text = doc_data['text']
        doc_path_segments = doc_data['path_segments']
        
        # Determine the true immediate child label for this document
        relative_to_root = os.path.relpath(current_data_node_path, root_data_dir)
        current_node_depth = len(relative_to_root.split(os.sep)) if relative_to_root != '.' else 0
        
        true_immediate_child_name = None
        if len(doc_path_segments) > current_node_depth:
            true_immediate_child_name = doc_path_segments[current_node_depth]

        # Skip documents that are in underscore-prefixed folders
        if true_immediate_child_name and true_immediate_child_name.startswith('_'):
            print(f"    Skipping document {os.path.basename(full_doc_path)}: In underscore-prefixed folder.")
            continue

        if true_immediate_child_name is None:
            print(f"    Warning: Skipping document {os.path.basename(full_doc_path)} due to invalid path segments.")
            continue

        # 1. Get aggregated output from current node's Folder CNN
        current_cnn_windows = _get_windows_from_text(folder_cnn_tokenizer, doc_text, current_cnn_window_size, current_cnn_stride)
        if not current_cnn_windows:
            print(f"    Warning: Document {os.path.basename(full_doc_path)} produced no windows for current CNN. Skipping DN sample.")
            continue

        current_cnn_window_predictions = folder_cnn_model.predict(np.array(current_cnn_windows), verbose=0)
        # Weighted average of window predictions
        current_cnn_aggregated_output = np.mean(current_cnn_window_predictions, axis=0) # Simple average for now, as sample_weights are for training

        # 2. Get aggregated outputs from child Folder CNNs (if applicable and model exists)
        current_dn_input_parts = [current_cnn_aggregated_output]
        
        for subfolder_name in immediate_subfolders:
            if subfolder_name in child_cnn_models: # If the child model was successfully loaded
                child_tokenizer = child_cnn_tokenizers[subfolder_name]
                child_cnn_model = child_cnn_models[subfolder_name]
                child_window_size = child_cnn_metadata_loaded[subfolder_name]['window_size']
                child_stride = child_cnn_metadata_loaded[subfolder_name]['stride']

                child_cnn_windows = _get_windows_from_text(child_tokenizer, doc_text, child_window_size, child_stride)
                if child_cnn_windows:
                    child_cnn_window_predictions = child_cnn_model.predict(np.array(child_cnn_windows), verbose=0)
                    child_cnn_aggregated_output = np.mean(child_cnn_window_predictions, axis=0)
                    current_dn_input_parts.append(child_cnn_aggregated_output)
                else:
                    print(f"    Warning: Document {os.path.basename(full_doc_path)} produced no windows for child CNN {subfolder_name}.")
                    current_dn_input_parts.append(np.zeros(expected_child_cnn_output_dims_for_metadata.get(subfolder_name, 0)))
            else:
                # Pad with zeros for missing or failed-to-load child CNN output
                current_dn_input_parts.append(np.zeros(expected_child_cnn_output_dims_for_metadata.get(subfolder_name, 0)))

        dn_inputs.append(np.concatenate(current_dn_input_parts))
        documents_processed_for_dn += 1

        # Determine Decision Network label
        # If the document's true path goes into an immediate subfolder: pass_to_SubX
        # If the document's true path ends at this level (i.e., it's a direct document in current folder): retain_at_current_level
        if true_immediate_child_name in immediate_subfolders:
            dn_labels.append(f'pass_to_{true_immediate_child_name}')
        else:
            dn_labels.append('retain_at_current_level') # Generic label for retaining at the current folder level
        print(f"    Prepared DN sample for {os.path.basename(full_doc_path)} with label: {dn_labels[-1]}")

    if not dn_inputs:
        print(f"  No suitable data for Decision Network for {current_data_node_path}.")
        return None, None, None, None

    dn_inputs_np = np.array(dn_inputs)
    dn_label_encoder = LabelEncoder()
    dn_integer_labels = dn_label_encoder.fit_transform(dn_labels)
    dn_one_hot_labels = tf.keras.utils.to_categorical(integer_labels, num_classes=len(dn_label_encoder.classes_))

    # Store the expected input dimension for prediction
    dn_input_shape_metadata = {
        'input_dim': dn_inputs_np.shape[1],
        'current_cnn_output_dim': current_cnn_output_dim,
        'child_cnn_output_dims': expected_child_cnn_output_dims_for_metadata # Store expected dims for all children
    }
    print(f"  Finished preparing {len(dn_inputs_np)} samples for Decision Network at {current_data_node_path} from {documents_processed_for_dn} documents.")
    return dn_inputs_np, dn_one_hot_labels, dn_label_encoder, dn_input_shape_metadata


def _train_decision_network_for_node(current_data_node_path, current_model_node_path, root_data_dir, model_base_dir, all_documents_data, args):
    """
    Trains the Decision Network for a specific node.
    Args:
        current_data_node_path (str): Path to the current data directory.
        current_model_node_path (str): Path to the current model directory.
        root_data_dir (str): The absolute root of the original document hierarchy.
        model_base_dir (str): The absolute base directory where all models are stored.
        all_documents_data (dict): Dictionary of all documents and their paths.
        args (argparse.Namespace): Command-line arguments.
    """
    print(f"  Training Decision Network for: {current_data_node_path}")

    # A Decision Network is only needed if there are actual subfolders to 'pass_to'
    # Filter out subfolders starting with an underscore
    subfolders = [d for d in os.listdir(current_data_node_path) if os.path.isdir(os.path.join(current_data_node_path, d)) and not d.startswith('_')]
    if not subfolders:
        print(f"  Skipping Decision Network for {current_data_node_path}: No non-underscore subfolders to pass to.")
        return

    dn_inputs, dn_labels_one_hot, dn_label_encoder, dn_input_shape_metadata = _prepare_decision_network_data(
        current_data_node_path, current_model_node_path, root_data_dir, model_base_dir, all_documents_data, args
    )

    if dn_inputs is None: # Preparation failed or no data
        return

    num_dn_input_nodes = dn_inputs.shape[1]
    num_dn_classes = len(dn_label_encoder.classes_)

    if num_dn_classes < 2:
        print(f"  Skipping Decision Network for {current_data_node_path}: Insufficient decision classes ({num_dn_classes}).")
        return

    dn_model = build_decision_network(num_dn_input_nodes, num_dn_classes, args.dropout_rate)
    dn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])

    X_dn_train, X_dn_val, y_dn_train, y_dn_val = train_test_split(dn_inputs, dn_labels_one_hot, test_size=0.2, random_state=42)
    print(f"    Training Decision Network for {current_data_node_path} with {len(X_dn_train)} training samples and {len(X_dn_val)} validation samples...")
    print(f"    Model Summary for Decision Network at {current_data_node_path}:")
    dn_model.summary(print_fn=print) # Print model summary

    dn_model.fit(X_dn_train, y_dn_train, epochs=args.epochs, batch_size=args.batch_size, validation_data=(X_dn_val, y_dn_val), verbose=1) # Changed verbose to 1
    
    loss, accuracy = dn_model.evaluate(X_dn_val, y_dn_val, verbose=0)
    print(f"    Decision Network Validation Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")


    # Save Decision Network, its label encoder, and input shape metadata
    os.makedirs(current_model_node_path, exist_ok=True)
    dn_model.save(os.path.join(current_model_node_path, 'decision_model.keras'))
    with open(os.path.join(current_model_node_path, 'decision_label_encoder_classes.json'), 'w') as f:
        json.dump(list(dn_label_encoder.classes_), f)
    with open(os.path.join(current_model_node_path, 'decision_input_shape.json'), 'w') as f:
        json.dump(dn_input_shape_metadata, f)
    print(f"  Decision Network saved to: {current_model_node_path}")


def train_hierarchical_models(args):
    """
    Orchestrates the training of all hierarchical CNNs and Decision Networks.
    """
    print(f"Starting hierarchical model training from root: {args.root_input_dir}")
    os.makedirs(args.model_output_base_dir, exist_ok=True)

    all_documents_data = _collect_all_documents_with_paths(args.root_input_dir)
    if not all_documents_data:
        print("No documents found in the root input directory. Exiting.")
        return

    # Phase 1: Train all Folder CNNs
    # Collect all directories that exist in the data hierarchy
    all_data_dirs = []
    # Use os.walk with filtering for underscore folders directly
    for root, dirs, files in os.walk(args.root_input_dir):
        # Filter out directories starting with an underscore for this traversal as well
        dirs[:] = [d for d in dirs if not d.startswith('_')]
        all_data_dirs.append(root)
    
    print("\n--- Phase 1: Training Folder CNNs ---")
    # Iterate through all directories to train their respective Folder CNNs
    for current_data_node_path in all_data_dirs:
        # Ensure the current_data_node_path itself doesn't start with an underscore (if it's not the root)
        if os.path.basename(current_data_node_path).startswith('_') and current_data_node_path != args.root_input_dir:
            print(f"  Skipping Folder CNN for {current_data_node_path}: Directory starts with an underscore.")
            continue

        relative_path_segment = os.path.relpath(current_data_node_path, args.root_input_dir)
        current_model_node_path = os.path.join(args.model_output_base_dir, relative_path_segment)
        
        _train_folder_cnn_for_node(
            current_data_node_path, current_model_node_path, all_documents_data, args
        )

    # Phase 2: Train all Decision Networks (bottom-up traversal to ensure child CNNs are available)
    print("\n--- Phase 2: Training Decision Networks ---")
    
    # Sort directories by depth in reverse order (deepest first)
    # Re-collect all_data_dirs to ensure it reflects the filtered structure
    all_data_dirs = []
    for root, dirs, files in os.walk(args.root_input_dir):
        dirs[:] = [d for d in dirs if not d.startswith('_')] # Filter again for consistency
        all_data_dirs.append(root)
    
    all_data_dirs.sort(key=lambda x: x.count(os.sep), reverse=True) 

    for current_data_node_path in all_data_dirs:
        # Ensure the current_data_node_path itself doesn't start with an underscore (if it's not the root)
        if os.path.basename(current_data_node_path).startswith('_') and current_data_node_path != args.root_input_dir:
            continue # Already skipped in Phase 1, just ensure consistency

        # Only train Decision Network if it has subfolders (non-underscore ones)
        subfolders = [d for d in os.listdir(current_data_node_path) if os.path.isdir(os.path.join(current_data_node_path, d)) and not d.startswith('_')]
        
        if subfolders: # If it has non-underscore subfolders, it might need a Decision Network
            relative_path_segment = os.path.relpath(current_data_node_path, args.root_input_dir)
            current_model_node_path = os.path.join(args.model_output_base_dir, relative_path_segment)
            
            # Ensure the Folder CNN for this node was successfully trained before attempting DN
            if os.path.exists(os.path.join(current_model_node_path, 'model.keras')):
                _train_decision_network_for_node(
                    current_data_node_path, current_model_node_path, args.root_input_dir, args.model_output_base_dir, all_documents_data, args
                )
            else:
                print(f"  Skipping Decision Network for {current_data_node_path}: Folder CNN not found or trained.")
        else:
            print(f"  Skipping Decision Network for {current_data_node_path}: No non-underscore subfolders.")


# --- Prediction Function ---
def _predict_single_document_hierarchically(doc_text, model_base_dir, current_model_node_path, root_data_dir):
    """
    Recursively predicts the hierarchical category of a single document.
    Args:
        doc_text (str): The content of the document to classify.
        model_base_dir (str): The base directory where all hierarchical models are stored.
        current_model_node_path (str): The current model directory being processed in the recursion.
        root_data_dir (str): The original root directory of the document hierarchy (for path validation).
    Returns:
        list: A list of strings representing the predicted hierarchical path segments.
    """
    # Load current node's Folder CNN and its assets
    folder_cnn_model_path = os.path.join(current_model_node_path, 'model.keras')
    folder_cnn_tokenizer_path = os.path.join(current_model_node_path, 'tokenizer_config.json')
    folder_cnn_label_classes_path = os.path.join(current_model_node_path, 'label_encoder_classes.json')
    folder_cnn_metadata_path = os.path.join(current_model_node_path, 'cnn_metadata.json')


    # If Folder CNN model doesn't exist, it means we've reached a leaf node (no further CNNs)
    # or an invalid path. The last successfully predicted category is the final one.
    if not os.path.exists(folder_cnn_model_path):
        return [os.path.basename(current_model_node_path)] 

    try:
        folder_cnn_model = load_model(folder_cnn_model_path)
        with open(folder_cnn_tokenizer_path, 'r') as f:
            folder_cnn_tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(json.load(f))
        with open(folder_cnn_label_classes_path, 'r') as f:
            folder_cnn_label_classes = json.load(f)
            folder_cnn_label_encoder = LabelEncoder()
            folder_cnn_label_encoder.classes_ = np.array(folder_cnn_label_classes)
        
        # Load window_size and stride for this specific CNN
        with open(folder_cnn_metadata_path, 'r') as f:
            folder_cnn_metadata = json.load(f)
        current_cnn_window_size = folder_cnn_metadata['window_size']
        current_cnn_stride = folder_cnn_metadata['stride']
        # Load filters and kernel_size (needed if model architecture changes based on these)
        current_cnn_filters = folder_cnn_metadata.get('filters', DEFAULT_CNN_FILTERS)
        current_cnn_kernel_size = folder_cnn_metadata.get('kernel_size', DEFAULT_CNN_KERNEL_SIZE)


    except Exception as e:
        print(f"Error loading Folder CNN for {current_model_node_path}: {e}")
        return [os.path.basename(current_model_node_path) + "_CNN_LOAD_ERROR"]

    # Get windows for the document and predict with current CNN
    current_cnn_windows = _get_windows_from_text(folder_cnn_tokenizer, doc_text, current_cnn_window_size, current_cnn_stride)
    if not current_cnn_windows:
        print(f"Warning: Document produced no valid windows for current CNN at {current_model_node_path}. Cannot predict.")
        return [os.path.basename(current_model_node_path) + "_NO_WINDOWS"]

    current_cnn_window_predictions = folder_cnn_model.predict(np.array(current_cnn_windows), verbose=0)
    # Aggregate window predictions (simple average as weights are for training)
    current_cnn_aggregated_output = np.mean(current_cnn_window_predictions, axis=0)


    # Check for Decision Network existence (meaning this is a non-leaf node with a decision logic)
    decision_model_path = os.path.join(current_model_node_path, 'decision_model.keras')
    decision_label_classes_path = os.path.join(current_model_node_path, 'decision_label_encoder_classes.json')
    decision_input_shape_path = os.path.join(current_model_node_path, 'decision_input_shape.json')

    if os.path.exists(decision_model_path):
        # This node has a Decision Network
        try:
            decision_model = load_model(decision_model_path)
            with open(decision_label_classes_path, 'r') as f:
                decision_label_classes = json.load(f)
                decision_label_encoder = LabelEncoder()
                decision_label_encoder.classes_ = np.array(decision_label_classes)
            with open(decision_input_shape_path, 'r') as f:
                dn_input_shape_metadata = json.load(f)
            
            expected_dn_input_dim = dn_input_shape_metadata['input_dim']
            expected_current_cnn_output_dim = dn_input_shape_metadata['current_cnn_output_dim']
            expected_child_cnn_output_dims = dn_input_shape_metadata['child_cnn_output_dims'] # This dictionary now stores the expected dims for each child

        except Exception as e:
            print(f"Error loading Decision Network for {current_model_node_path}: {e}")
            return [os.path.basename(current_model_node_path) + "_DN_LOAD_ERROR"]

        # Gather inputs for the Decision Network
        dn_inputs_list = [current_cnn_aggregated_output] # Start with current CNN's aggregated output

        # Get aggregated outputs from child Folder CNNs (if they exist and have models)
        # Iterate through all *expected* children from metadata to ensure consistent input shape
        for child_dir_name, child_expected_output_dim in expected_child_cnn_output_dims.items():
            # Only consider child directories that do NOT start with an underscore
            if child_dir_name.startswith('_'):
                continue # Skip underscore-prefixed child directories

            child_model_dir = os.path.join(current_model_node_path, child_dir_name)
            child_model_path = os.path.join(child_model_dir, 'model.keras')
            child_tokenizer_path = os.path.join(child_model_dir, 'tokenizer_config.json')
            child_metadata_path = os.path.join(child_model_dir, 'cnn_metadata.json')

            child_cnn_aggregated_output = np.zeros(child_expected_output_dim) # Default to zeros if model not found/loaded
            if os.path.exists(child_model_path) and os.path.exists(child_tokenizer_path) and os.path.exists(child_metadata_path):
                try:
                    child_cnn_model = load_model(child_model_path)
                    with open(child_tokenizer_path, 'r') as f:
                        child_cnn_tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(json.load(f))
                    with open(child_metadata_path, 'r') as f:
                        child_cnn_metadata = json.load(f)
                    child_window_size = child_cnn_metadata['window_size']
                    child_stride = child_cnn_metadata['stride']

                    child_cnn_windows = _get_windows_from_text(child_cnn_tokenizer, doc_text, child_window_size, child_stride)
                    if child_cnn_windows:
                        child_cnn_window_predictions = child_cnn_model.predict(np.array(child_cnn_windows), verbose=0)
                        child_cnn_aggregated_output = np.mean(child_cnn_window_predictions, axis=0)
                    else:
                        print(f"Warning: Document produced no valid windows for child CNN {child_dir_name} at {current_model_node_path}.")
                except Exception as e:
                    print(f"Warning: Could not load or run child CNN for {child_dir_name} at {current_model_node_path}: {e}")
            dn_inputs_list.append(child_cnn_aggregated_output)

        # Concatenate all inputs for the Decision Network
        dn_input_array = np.concatenate(dn_inputs_list).reshape(1, -1)

        # Ensure the input array matches the expected dimension from training
        if dn_input_array.shape[1] != expected_dn_input_dim:
            print(f"Error: Decision Network input shape mismatch. Expected {expected_dn_input_dim}, got {dn_input_array.shape[1]}. Adjusting.")
            if dn_input_array.shape[1] < expected_dn_input_dim:
                dn_input_array = np.pad(dn_input_array, ((0,0), (0, expected_dn_input_dim - dn_input_array.shape[1])), 'constant')
            elif dn_input_array.shape[1] > expected_dn_input_dim:
                dn_input_array = dn_input_array[:, :expected_dn_input_dim]

        decision_probabilities = decision_model.predict(dn_input_array, verbose=0)[0]
        chosen_decision_index = np.argmax(decision_probabilities)
        chosen_decision_label = decision_label_encoder.inverse_transform([chosen_decision_index])[0]

        # Act based on the decision
        if chosen_decision_label.startswith('pass_to_'):
            next_category_name = chosen_decision_label.replace('pass_to_', '')
            next_model_node_path = os.path.join(current_model_node_path, next_category_name)
            
            # Validate if the target is an actual subdirectory in the original data structure
            # This prevents trying to recurse into non-existent paths or direct documents
            relative_current_data_path = os.path.relpath(current_model_node_path, model_base_dir)
            potential_next_data_path = os.path.join(root_data_dir, relative_current_data_path, next_category_name)

            if os.path.isdir(potential_next_data_path) and not next_category_name.startswith('_'): # Ensure next folder is not underscore-prefixed
                return [os.path.basename(current_model_node_path)] + _predict_single_document_hierarchically(
                    doc_text, model_base_dir, next_model_node_path, root_data_dir
                )
            else:
                # Decision was to pass, but target is not a valid non-underscore subfolder in data, retain at current level
                return [os.path.basename(current_model_node_path), f"RETAIN_FALLBACK_INVALID_PASS_TO_{next_category_name}"]

        elif chosen_decision_label == 'retain_at_current_level': # Check for the generic retain label
            # It's a final classification at this level, so the document belongs to the current folder.
            return [os.path.basename(current_model_node_path)]
        else:
            # Fallback for unexpected decision label (shouldn't happen with proper training)
            return [os.path.basename(current_model_node_path) + "_UNKNOWN_DECISION"]

    else:
        # No Decision Network means it's a leaf node for this type of hierarchy,
        # or a direct document at this level without further subfolders.
        # The 'final' classification is based on the Folder CNN's prediction.
        predicted_class_index = np.argmax(current_cnn_aggregated_output)
        predicted_category = folder_cnn_label_encoder.inverse_transform([predicted_class_index])[0]
        return [os.path.basename(current_model_node_path), predicted_category] # Current folder + predicted category


def predict_hierarchical_documents(args):
    """
    Orchestrates hierarchical prediction for one or more documents.
    """
    print(f"Starting hierarchical prediction from model base: {args.model_base_dir}")

    documents_to_predict = []
    document_names = []

    if args.input_file:
        try:
            doc_content = get_document_text(args.input_file) # Use get_document_text here
            documents_to_predict.append(doc_content)
            document_names.append(os.path.basename(args.input_file))
        except (ValueError, FileNotFoundError, IOError) as e:
            print(f"Error reading input file {args.input_file}: {e}")
            return
    elif args.input_dir:
        for doc_file_name in os.listdir(args.input_dir):
            file_path = os.path.join(args.input_dir, doc_file_name)
            if os.path.isfile(file_path):
                try:
                    doc_content = get_document_text(file_path) # Use get_document_text here
                    documents_to_predict.append(doc_content)
                    document_names.append(doc_file_name)
                except (ValueError, FileNotFoundError, IOError) as e:
                    print(f"Warning: Could not read document {file_path}: {e}")
    elif args.raw_text:
        documents_to_predict.append(args.raw_text)
        document_names.append("raw_text_input")
    else:
        print("Please provide --input-file, --input-dir, or --raw-text for prediction.")
        return

    if not documents_to_predict:
        print("No documents found for prediction.")
        return

    results = []
    for i, doc_text in enumerate(documents_to_predict):
        predicted_path_segments = _predict_single_document_hierarchically(
            doc_text, args.model_base_dir, args.model_base_dir, args.root_input_dir
        )
        full_path_str = " -> ".join(predicted_path_segments)
        results.append(f"Document: {document_names[i]}, Predicted Path: {full_path_str}")
        print(results[-1])

    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for i, doc_name in enumerate(document_names):
                path_segments = _predict_single_document_hierarchically(
                    documents_to_predict[i], args.model_base_dir, args.model_base_dir, args.root_input_dir
                )
                f.write(f"{doc_name},{'/'.join(path_segments)}\n")
        print(f"Predictions saved to {args.output_file}")


def main():
    """Main function to parse arguments and execute commands."""
    parser = argparse.ArgumentParser(description="Command-line utility for hierarchical document categorization using TensorFlow.")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- Train Hierarchical Subparser ---
    train_hierarchical_parser = subparsers.add_parser("train-hierarchical", help="Train hierarchical document categorization models.")
    train_hierarchical_parser.add_argument("--root-input-dir", required=True, help="Root directory containing hierarchical documents.")
    train_hierarchical_parser.add_argument("--model-output-base-dir", required=True, help="Base directory to save all hierarchical TensorFlow models.")
    train_hierarchical_parser.add_argument("--max-vocab-size", type=int, default=DEFAULT_MAX_VOCAB_SIZE, help="Maximum number of words in vocabulary for all models.")
    train_hierarchical_parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE, 
                                           help="Size of the text window (sequence length) for CNN input.")
    train_hierarchical_parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE,
                                           help="Stride for moving the window (overlap if stride < window_size).")
    train_hierarchical_parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs for each CNN.")
    train_hierarchical_parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for training each CNN.")
    train_hierarchical_parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE, help="Learning rate for the optimizer.")
    train_hierarchical_parser.add_argument("--embedding-dim", type=int, default=DEFAULT_EMBEDDING_DIM, help="Dimension of the word embeddings.")
    train_hierarchical_parser.add_argument("--dropout-rate", type=float, default=DEFAULT_DROPOUT_RATE, help="Dropout rate for regularization.")
    train_hierarchical_parser.add_argument("--cnn-filters", type=int, default=DEFAULT_CNN_FILTERS, help="Number of filters for the CNN's Conv1D layer.")
    train_hierarchical_parser.add_argument("--cnn-kernel-size", type=int, default=DEFAULT_CNN_KERNEL_SIZE, help="Kernel size for the CNN's Conv1D layer.")
    train_hierarchical_parser.set_defaults(func=train_hierarchical_models)

    # --- Predict Hierarchical Subparser ---
    predict_hierarchical_parser = subparsers.add_parser("predict-hierarchical", help="Predict categories for new documents using hierarchical models.")
    predict_hierarchical_parser.add_argument("--model-base-dir", required=True, help="Base directory where all hierarchical TensorFlow models are stored.")
    predict_hierarchical_parser.add_argument("--root-input-dir", required=True, help="Original root directory of documents (needed to reconstruct data structure for prediction logic).")
    predict_hierarchical_parser.add_argument("--input-file", help="Path to a single document file for prediction.")
    predict_hierarchical_parser.add_argument("--input-dir", help="Directory containing multiple document files for prediction.")
    predict_hierarchical_parser.add_argument("--raw-text", help="Provide raw text directly as input.")
    predict_hierarchical_parser.add_argument("--output-file", help="Path to save prediction results (e.g., CSV).")
    predict_hierarchical_parser.set_defaults(func=predict_hierarchical_documents)

    args = parser.parse_args()

    # Execute the function associated with the chosen command
    if hasattr(args, 'func'):
        args.func(args)
    else:
        # If no command is provided, print help message
        parser.print_help()

if __name__ == "__main__":
    main()

