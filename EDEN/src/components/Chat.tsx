import React, { useState } from 'react';

interface ChatMessage {
    id: string;
    name: string;
    text: string;
}

const Chat = ({ messages, onSend, selectedTarget }: { messages: ChatMessage[], onSend: (text: string, tab: 'player' | 'ai') => void, selectedTarget?: string | null }) => {
    const [input, setInput] = useState('');
    const [activeTab, setActiveTab] = useState<'player' | 'ai'>('player');

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (input.trim()) {
            onSend(input, activeTab);
            setInput('');
        }
    };

    // Filter messages logic
    const filteredMessages = messages.filter(msg => {
        if (activeTab === 'player') return msg.id !== 'SYSTEM';
        if (activeTab === 'ai') return msg.id === 'SYSTEM';
        return true;
    });

    const tabStyle = (tab: 'player' | 'ai') => ({
        flex: 1,
        padding: '10px',
        textAlign: 'center' as const,
        cursor: 'pointer',
        background: activeTab === tab ? '#444' : 'transparent',
        borderBottom: activeTab === tab ? '2px solid #00ccff' : '1px solid #444',
        color: activeTab === tab ? 'white' : '#aaa',
        fontWeight: activeTab === tab ? 'bold' : 'normal',
    });

    return (
        <div
            onClick={(e) => e.stopPropagation()}
            onMouseDown={(e) => e.stopPropagation()}
            style={{
                position: 'absolute',
                bottom: '20px',
                right: '20px',
                width: '350px',
                height: '450px',
                background: 'rgba(0, 0, 0, 0.9)',
                borderRadius: '10px',
                display: 'flex',
                flexDirection: 'column',
                fontFamily: 'sans-serif',
                fontSize: '14px',
                color: 'white',
                zIndex: 1000,
                boxShadow: '0 4px 10px rgba(0,0,0,0.5)'
            }}>
            <div style={{ display: 'flex', borderBottom: '1px solid #444' }}>
                <div onClick={() => setActiveTab('player')} style={tabStyle('player')}>
                    Player Chat
                </div>
                <div onClick={() => setActiveTab('ai')} style={tabStyle('ai')}>
                    AI Builder
                </div>
            </div>

            {/* Target Display in AI Tab */}
            {activeTab === 'ai' && (
                <div style={{ padding: '5px 10px', background: '#333', fontSize: '12px', color: '#ffcc00', borderBottom: '1px solid #555' }}>
                    Target: {selectedTarget || 'None (New Object Mode)'}
                </div>
            )}

            <div style={{ flex: 1, overflowY: 'auto', padding: '10px' }}>
                {filteredMessages.length === 0 && (
                    <div style={{ color: '#666', fontStyle: 'italic', textAlign: 'center', marginTop: '20px' }}>
                        {activeTab === 'ai'
                            ? (selectedTarget ? 'Type a behavior for this object...' : 'No AI logs yet. Try creating something!')
                            : 'No messages yet.'}
                    </div>
                )}
                {filteredMessages.map((msg, idx) => (
                    <div key={idx} style={{ marginBottom: '8px', wordWrap: 'break-word', lineHeight: '1.4' }}>
                        <span style={{ color: msg.name === 'EDEN AI' ? '#00ccff' : msg.name === 'Guest' ? '#aaa' : '#ffcc00', fontWeight: 'bold' }}>
                            {msg.name}:
                        </span>{' '}
                        <span style={{ color: msg.id === 'SYSTEM' ? '#ddd' : 'white' }}>
                            {msg.text}
                        </span>
                    </div>
                ))}
            </div>
            <form onSubmit={handleSubmit} style={{ padding: '10px', display: 'flex', borderTop: '1px solid #444' }}>
                <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder={activeTab === 'ai' ? (selectedTarget ? "ex: Jump every 5s" : "ex: Create a red sphere") : "Say hello..."}
                    style={{ flex: 1, padding: '10px', borderRadius: '4px', border: 'none', marginRight: '5px', background: '#222', color: 'white' }}
                />
                <button type="submit" style={{ padding: '10px 15px', background: activeTab === 'ai' ? '#9900ff' : '#0066cc', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer', fontWeight: 'bold' }}>
                    Send
                </button>
            </form>
        </div>
    );
};
export default Chat;
