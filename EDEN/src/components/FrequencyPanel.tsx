import React, { useState } from 'react';
import SliderRow from './SliderRow';

interface FrequencyPanelProps {
    frequency: number;
    onChange: (frequency: number) => void;
}

const FrequencyPanel: React.FC<FrequencyPanelProps> = ({ frequency, onChange }) => {
    const [isExpanded, setIsExpanded] = useState(false);

    if (!isExpanded) {
        return (
            <button
                onClick={() => setIsExpanded(true)}
                style={{
                    position: 'absolute',
                    top: '80px', // Position below the top-right buttons
                    right: '20px',
                    background: 'rgba(0, 0, 0, 0.6)',
                    color: '#fff',
                    border: '1px solid #444',
                    borderRadius: '8px',
                    padding: '8px 12px',
                    cursor: 'pointer',
                    zIndex: 2000,
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px',
                    backdropFilter: 'blur(4px)',
                }}
            >
                📡 <span style={{ fontSize: '12px' }}>Freq: {frequency}</span>
            </button>
        );
    }

    return (
        <div
            style={{
                position: 'absolute',
                top: '80px',
                right: '20px',
                width: '240px',
                background: 'rgba(20, 20, 20, 0.9)',
                borderRadius: '12px',
                padding: '16px',
                boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
                border: '1px solid #333',
                zIndex: 2000,
                backdropFilter: 'blur(8px)',
                color: '#fff',
            }}
            onClick={(e) => e.stopPropagation()}
        >
            <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginBottom: '16px',
                borderBottom: '1px solid #333',
                paddingBottom: '8px'
            }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{ fontSize: '18px' }}>📡</span>
                    <span style={{ fontWeight: 600, fontSize: '14px', color: '#eee' }}>Frequency</span>
                </div>
                <button
                    onClick={() => setIsExpanded(false)}
                    style={{
                        background: 'transparent',
                        border: 'none',
                        color: '#666',
                        cursor: 'pointer',
                        fontSize: '18px',
                        padding: '0 4px',
                        lineHeight: 1,
                    }}
                >
                    −
                </button>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                <SliderRow
                    label="Channel"
                    value={frequency}
                    min={0}
                    max={100}
                    step={1}
                    onChange={onChange}
                />
            </div>

            <div style={{
                marginTop: '12px',
                fontSize: '11px',
                color: '#666',
                textAlign: 'center',
                fontStyle: 'italic'
            }}>
                Adjust frequency to switch world layers
            </div>
        </div>
    );
};

export default FrequencyPanel;
