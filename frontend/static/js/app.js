class MultimodalRAGInterface {
    constructor() {
        this.chatMessages = document.getElementById('chatMessages');
        this.textInput = document.getElementById('textInput');
        this.fileInput = document.getElementById('fileInput');
        this.sendButton = document.getElementById('sendButton');
        this.filePreview = document.getElementById('filePreview');
        this.previewImage = document.getElementById('previewImage');
        this.fileName = document.getElementById('fileName');
        
        this.currentFile = null;
        this.isLoading = false;
        this.apiBaseUrl = '/api';
        
        this.initEventListeners();
    }
    
    initEventListeners() {
        this.sendButton.addEventListener('click', () => this.sendMessage());
        this.textInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        this.fileInput.addEventListener('change', (e) => this.handleFileUpload(e));
        
        // Auto-resize textarea
        this.textInput.addEventListener('input', () => {
            this.textInput.style.height = 'auto';
            this.textInput.style.height = Math.min(this.textInput.scrollHeight, 120) + 'px';
        });
    }
    
    handleFileUpload(event) {
        const file = event.target.files[0];
        if (file) {
            this.currentFile = file;
            const reader = new FileReader();
            reader.onload = (e) => {
                this.previewImage.src = e.target.result;
                this.fileName.textContent = file.name;
                this.filePreview.classList.add('active');
            };
            reader.readAsDataURL(file);
        }
    }
    
    async sendMessage() {
        const text = this.textInput.value.trim();
        if (!text && !this.currentFile) return;
        
        if (this.isLoading) return;
        
        this.isLoading = true;
        this.sendButton.disabled = true;
        
        // Add user message
        this.addMessage('user', text, this.currentFile);
        
        // Clear input
        this.textInput.value = '';
        this.textInput.style.height = 'auto';
        this.clearFilePreview();
        
        // Show loading
        const loadingMessage = this.addMessage('bot', '', null, true);
        
        try {
            // Call actual API
            const response = await this.callRAGAPI(text, this.currentFile);
            
            // Remove loading message
            loadingMessage.remove();
            
            if (response.success) {
                // Add bot response
                this.addMessage('bot', response.answer, null, false, response.search_results);
            } else {
                this.addMessage('bot', `Error: ${response.error}`, null, false);
            }
            
        } catch (error) {
            loadingMessage.remove();
            this.addMessage('bot', `Network error: ${error.message}`, null, false);
        }
        
        this.isLoading = false;
        this.sendButton.disabled = false;
        this.currentFile = null;
    }
    
    addMessage(sender, text, file = null, isLoading = false, searchResults = null) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;
        
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.textContent = sender === 'user' ? 'U' : 'AI';
        
        const content = document.createElement('div');
        content.className = 'message-content';
        
        if (isLoading) {
            content.innerHTML = `
                <div class="loading">
                    <div class="loading-spinner"></div>
                    <span>Searching and analyzing...</span>
                </div>
            `;
        } else {
            // content.innerHTML = `<div>${text}</div>`;
            content.innerHTML = `<div>${marked.parse(text)}</div>`;

            
            if (file) {
                const img = document.createElement('img');
                img.src = URL.createObjectURL(file);
                img.className = 'uploaded-image';
                content.appendChild(img);
            }
            
            if (searchResults && searchResults.length > 0) {
                const resultsDiv = document.createElement('div');
                resultsDiv.className = 'search-results';
                resultsDiv.innerHTML = '<h4>📚 Retrieved Sources:</h4>';
                
                searchResults.forEach(result => {
                    const resultItem = document.createElement('div');
                    resultItem.className = 'search-result-item';
                    resultItem.innerHTML = `
                        <div class="search-result-content">
                            <div class="search-result-title">${result.title}</div>
                            <div class="search-result-snippet">${result.snippet}</div>
                        </div>
                        <div class="search-result-meta">
                            <div class="similarity-score">${(result.score * 100).toFixed(1)}%</div>
                            <a href="#" class="document-link" onclick="openDocument('${result.source}', ${result.page})">
                                ${result.source} - Page ${result.page}
                            </a>
                        </div>
                    `;
                    resultsDiv.appendChild(resultItem);
                });
                
                content.appendChild(resultsDiv);
            }
        }
        
        messageDiv.appendChild(avatar);
        messageDiv.appendChild(content);
        
        // Remove empty state if it exists
        const emptyState = this.chatMessages.querySelector('.empty-state');
        if (emptyState) {
            emptyState.remove();
        }
        
        this.chatMessages.appendChild(messageDiv);
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
        
        return messageDiv;
    }
    
    clearFilePreview() {
        this.filePreview.classList.remove('active');
        this.previewImage.src = '';
        this.fileName.textContent = '';
        this.fileInput.value = '';
    }
    
    async callRAGAPI(text, file) {
        const formData = new FormData();
        
        // Prepare request data
        const requestData = {
            text: text
        };
        
        if (file) {
            // Convert file to base64
            const base64 = await this.fileToBase64(file);
            requestData.image = base64;
        }
        
        const response = await fetch(`${this.apiBaseUrl}/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestData)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    }
    
    fileToBase64(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.readAsDataURL(file);
            reader.onload = () => resolve(reader.result);
            reader.onerror = error => reject(error);
        });
    }
}

// Global function for document links
function openDocument(source, page) {
    // In a real implementation, this would open the document viewer
    // For now, we'll try to open the document via API
    window.open(`/api/document/${source}#page=${page}`, '_blank');
}

// Initialize the interface when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new MultimodalRAGInterface();
});