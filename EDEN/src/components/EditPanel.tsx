import React, { useState, useEffect } from 'react';

interface EditPanelProps {
    type: 'color' | 'size';
    currentValue: string | number[];
    targetName: string;
    onConfirm: (value: string | number[]) => void;
    onCancel: () => void;
}

const EditPanel: React.FC<EditPanelProps> = ({ type, currentValue, targetName, onConfirm, onCancel }) => {
    const [colorValue, setColorValue] = useState(typeof currentValue === 'string' ? currentValue : '#ff0000');
    const [sizeValue, setSizeValue] = useState(
        Array.isArray(currentValue) ? currentValue.join(', ') : '1, 1, 1'
    );
    const [isDirty, setIsDirty] = useState(false);

    useEffect(() => {
        if (isDirty) return;
        if (type === 'color' && typeof currentValue === 'string') {
            setColorValue(currentValue);
        } else if (type === 'size' && Array.isArray(currentValue)) {
            setSizeValue(currentValue.join(', '));
        }
    }, [type, currentValue, isDirty]);

    useEffect(() => {
        setIsDirty(false);
    }, [type, targetName]);

    const handleConfirm = () => {
        if (type === 'color') {
            onConfirm(colorValue);
        } else {
            const parsed = sizeValue.split(',').map(s => parseFloat(s.trim())).filter(n => !isNaN(n));
            if (parsed.length >= 1) {
                // Pad to 3 values if needed
                while (parsed.length < 3) parsed.push(parsed[parsed.length - 1] || 1);
                onConfirm(parsed);
            }
        }
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter') {
            handleConfirm();
        } else if (e.key === 'Escape') {
            onCancel();
        }
    };

    return (
        <div
            onClick={(e) => e.stopPropagation()}
            onMouseDown={(e) => e.stopPropagation()}
            style={{
                position: 'fixed',
                top: 0,
                left: 0,
                right: 0,
                bottom: 0,
                background: 'rgba(0, 0, 0, 0.7)',
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                zIndex: 3000,
            }}
        >
            <div
                style={{
                    background: '#1a1a1a',
                    borderRadius: '12px',
                    padding: '24px',
                    minWidth: '320px',
                    boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
                    border: '1px solid #333',
                }}
            >
                <h3 style={{
                    margin: '0 0 8px 0',
                    color: '#fff',
                    fontSize: '18px',
                    fontWeight: 600
                }}>
                    {type === 'color' ? '🎨 Change Color' : '📐 Change Size'}
                </h3>
                {/* Close Button */}
                <button
                    onClick={onCancel}
                    style={{
                        position: 'absolute',
                        top: '16px',
                        right: '16px',
                        background: 'transparent',
                        border: 'none',
                        color: '#666',
                        fontSize: '20px',
                        cursor: 'pointer',
                        padding: '4px',
                        lineHeight: 1,
                    }}
                >
                    ×
                </button>
                <p style={{
                    margin: '0 0 20px 0',
                    color: '#888',
                    fontSize: '13px'
                }}>
                    Editing: <span style={{ color: '#ffcc00' }}>{targetName}</span>
                </p>

                {type === 'color' ? (
                    <div style={{ marginBottom: '20px' }}>
                        <label style={{ display: 'block', color: '#aaa', fontSize: '12px', marginBottom: '8px' }}>
                            Color (hex)
                        </label>
                        <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
                            <input
                                type="color"
                                value={colorValue}
                                onChange={(e) => {
                                    setIsDirty(true);
                                    setColorValue(e.target.value);
                                }}
                                style={{
                                    width: '60px',
                                    height: '40px',
                                    border: 'none',
                                    borderRadius: '8px',
                                    cursor: 'pointer',
                                    background: 'transparent',
                                }}
                            />
                            <input
                                type="text"
                                value={colorValue}
                                onChange={(e) => {
                                    setIsDirty(true);
                                    setColorValue(e.target.value);
                                }}
                                onKeyDown={handleKeyDown}
                                autoFocus
                                style={{
                                    flex: 1,
                                    padding: '12px',
                                    borderRadius: '8px',
                                    border: '1px solid #444',
                                    background: '#222',
                                    color: '#fff',
                                    fontSize: '16px',
                                    fontFamily: 'monospace',
                                }}
                            />
                        </div>
                    </div>
                ) : (
                    <div style={{ marginBottom: '20px' }}>
                        <label style={{ display: 'block', color: '#aaa', fontSize: '12px', marginBottom: '8px' }}>
                            Size (width, height, depth)
                        </label>
                        <input
                            type="text"
                            value={sizeValue}
                            onChange={(e) => {
                                setIsDirty(true);
                                setSizeValue(e.target.value);
                            }}
                            onKeyDown={handleKeyDown}
                            autoFocus
                            placeholder="e.g., 2, 2, 2"
                            style={{
                                width: '100%',
                                padding: '12px',
                                borderRadius: '8px',
                                border: '1px solid #444',
                                background: '#222',
                                color: '#fff',
                                fontSize: '16px',
                                fontFamily: 'monospace',
                                boxSizing: 'border-box',
                            }}
                        />
                        <p style={{ margin: '8px 0 0 0', color: '#666', fontSize: '11px' }}>
                            Enter 1-3 comma-separated numbers
                        </p>
                    </div>
                )}

                <div style={{ display: 'flex', gap: '10px', justifyContent: 'flex-end' }}>
                    <button
                        onClick={onCancel}
                        style={{
                            padding: '10px 20px',
                            borderRadius: '8px',
                            border: '1px solid #444',
                            background: 'transparent',
                            color: '#aaa',
                            cursor: 'pointer',
                            fontSize: '14px',
                        }}
                    >
                        Cancel
                    </button>
                    <button
                        onClick={handleConfirm}
                        style={{
                            padding: '10px 20px',
                            borderRadius: '8px',
                            border: 'none',
                            background: type === 'color' ? '#9900ff' : '#00ccff',
                            color: '#fff',
                            cursor: 'pointer',
                            fontSize: '14px',
                            fontWeight: 600,
                        }}
                    >
                        Apply
                    </button>
                </div>
            </div>
        </div>
    );
};

export default EditPanel;
