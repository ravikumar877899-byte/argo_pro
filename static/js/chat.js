let currentLang = 'en';

function toggleChat() {
    const chatWindow = document.getElementById('chat-window');
    if (chatWindow.style.display === 'none' || chatWindow.style.display === '') {
        chatWindow.style.display = 'flex';
    } else {
        chatWindow.style.display = 'none';
    }
}

function changeLang() {
    currentLang = document.getElementById('chat-lang').value;
    const messages = document.getElementById('chat-messages');

    // Clear initial greeting if it's the only message and replace with localized one
    if (messages.children.length === 1) {
        messages.innerHTML = '';
        const greeting = currentLang === 'ta' ?
            "வணக்கம்! நான் உங்கள் அக்ரோஏஐ (AgroAI) உதவியாளர். பயிர் நோய்கள் அல்லது சிகிச்சையைப் பற்றி ஏதேனும் என்னிடம் கேளுங்கள்!" :
            "Hi! I'm your AI crop assistant. Ask me anything about plant diseases or treatments!";
        appendMessage('bot', greeting);
    }
}

document.getElementById('chat-toggle').addEventListener('click', toggleChat);

async function sendMessage() {
    const input = document.getElementById('chat-input');
    const message = input.value.trim();
    if (!message) return;

    appendMessage('user', message);
    input.value = '';

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: message, lang: currentLang })
        });
        const data = await response.json();
        appendMessage('bot', data.response);
    } catch (err) {
        console.error('Chat error:', err);
        const errorMsg = currentLang === 'ta' ?
            "மன்னிக்கவும், இப்போது இணைப்பதில் சிக்கல் உள்ளது. பின்னர் மீண்டும் முயற்சிக்கவும்." :
            "Sorry, I'm having trouble connecting to the brain right now. Please try again later.";
        appendMessage('bot', errorMsg);
    }
}

function appendMessage(sender, text) {
    const messages = document.getElementById('chat-messages');
    const div = document.createElement('div');

    const isUser = sender === 'user';
    div.style.alignSelf = isUser ? 'flex-end' : 'flex-start';
    div.style.background = isUser ? 'var(--primary-color)' : '#e8f5e9';
    div.style.color = isUser ? 'white' : 'var(--primary-dark)';
    div.style.padding = '10px 15px';
    div.style.borderRadius = isUser ? '15px 15px 0 15px' : '15px 15px 15px 0';
    div.style.maxWidth = '80%';
    div.style.fontSize = '0.95rem';
    div.style.border = isUser ? 'none' : '1px solid #c8e6c9';
    div.style.boxShadow = '0 2px 5px rgba(0,0,0,0.05)';

    div.textContent = text;
    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
}

document.getElementById('send-chat').addEventListener('click', sendMessage);
document.getElementById('chat-input').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendMessage();
});
