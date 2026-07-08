import React, { useState, useEffect, useRef } from 'react';

interface ChatMessage {
    id: string;
    name: string;
    text: string;
}

interface MobileChatProps {
    messages: ChatMessage[];
    onSend: (text: string, tab: 'player' | 'ai') => void;
    selectedTarget?: string | null;
    onClose: () => void;
}

const MobileChat: React.FC<MobileChatProps> = ({ messages, onSend, selectedTarget, onClose }) => {
    const [input, setInput] = useState('');
    const [activeTab, setActiveTab] = useState<'player' | 'ai'>('player');
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages, activeTab]);

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (input.trim()) {
            onSend(input, activeTab);
            setInput('');
        }
    };

    const filteredMessages = messages.filter(msg => {
        if (activeTab === 'player') return msg.id !== 'SYSTEM';
        if (activeTab === 'ai') return msg.id === 'SYSTEM';
        return true;
    });

    return (
        <div style={{
            position: 'fixed', bottom: 0, left: 0, width: '100%', height: '50%',
            background: 'rgba(0,0,0,0.95)', zIndex: 100000, display: 'flex', flexDirection: 'column'
        }}>
            {/* Header */}
            <div style={{
                padding: '12px 16px', background: '#222', display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                borderBottom: '1px solid #333'
            }}>
                <div style={{ display: 'flex', gap: '16px' }}>
                    <button
                        onClick={() => setActiveTab('player')}
                        style={{
                            background: 'transparent', border: 'none', color: activeTab === 'player' ? 'white' : '#666',
                            fontSize: '16px', fontWeight: activeTab === 'player' ? 'bold' : 'normal',
                            borderBottom: activeTab === 'player' ? '2px solid #00ccff' : 'none', paddingBottom: '4px'
                        }}
                    >
                        Chat
                    </button>
                    <button
                        onClick={() => setActiveTab('ai')}
                        style={{
                            background: 'transparent', border: 'none', color: activeTab === 'ai' ? 'white' : '#666',
                            fontSize: '16px', fontWeight: activeTab === 'ai' ? 'bold' : 'normal',
                            borderBottom: activeTab === 'ai' ? '2px solid #9900ff' : 'none', paddingBottom: '4px'
                        }}
                    >
                        AI Builder
                    </button>
                </div>
                <button
                    onClick={onClose}
                    style={{ background: 'transparent', border: 'none', color: '#888', fontSize: '24px', lineHeight: 1 }}
                >
                    ×
                </button>
            </div>

            {/* AI Target Info */}
            {activeTab === 'ai' && (
                <div style={{ padding: '8px 16px', background: '#333', fontSize: '13px', color: '#ffcc00' }}>
                    Target: {selectedTarget || 'World (Create Mode)'}
                </div>
            )}

            {/* Messages */}
            <div style={{ flex: 1, overflowY: 'auto', padding: '16px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
                {filteredMessages.length === 0 && (
                    <div style={{ color: '#555', textAlign: 'center', marginTop: '40px' }}>
                        No messages yet
                    </div>
                )}
                {filteredMessages.map((msg, idx) => (
                    <div key={idx} style={{
                        alignSelf: msg.name === 'Me' ? 'flex-end' : 'flex-start',
                        maxWidth: '85%',
                        background: msg.name === 'EDEN AI' ? 'rgba(153, 0, 255, 0.2)' : (msg.name === 'Me' ? '#004488' : '#333'),
                        padding: '8px 12px',
                        borderRadius: '12px',
                        borderBottomLeftRadius: msg.name !== 'Me' ? '2px' : '12px',
                        borderBottomRightRadius: msg.name === 'Me' ? '2px' : '12px',
                        border: msg.name === 'EDEN AI' ? '1px solid rgba(153, 0, 255, 0.4)' : 'none'
                    }}>
                        <div style={{ fontSize: '11px', color: '#aaa', marginBottom: '2px' }}>{msg.name}</div>
                        <div style={{ color: 'white', fontSize: '15px', lineHeight: '1.4' }}>{msg.text}</div>
                    </div>
                ))}
                <div ref={messagesEndRef} />
            </div>

            {/* Input */}
            <form onSubmit={handleSubmit} style={{ padding: '12px', background: '#222', display: 'flex', gap: '8px' }}>
                <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder={activeTab === 'ai' ? "Describe what to make..." : "Message..."}
                    style={{
                        flex: 1, padding: '12px', borderRadius: '24px', border: 'none', background: '#444', color: 'white', fontSize: '16px'
                    }}
                />
                <button
                    type="submit"
                    disabled={!input.trim()}
                    style={{
                        width: '48px', height: '48px', borderRadius: '50%', border: 'none',
                        background: input.trim() ? (activeTab === 'ai' ? '#9900ff' : '#00ccff') : '#555',
                        color: 'white', display: 'flex', alignItems: 'center', justifyContent: 'center',
                        fontSize: '20px'
                    }}
                >
                    ➤
                </button>
            </form>
        </div>
    );
};

export default MobileChat;
