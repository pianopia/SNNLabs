import React from 'react';

interface ContextMenuProps {
    x: number;
    y: number;
    targetName: string;
    targetInfo?: {
        shape?: string;
        color?: string;
        size?: number[];
        isNpc?: boolean;
    };
    onClose: () => void;
    onAction: (action: string, data?: any) => void;
}

const MenuButton = ({
    onClick,
    icon,
    label,
    danger = false
}: {
    onClick: () => void;
    icon: string;
    label: string;
    danger?: boolean;
}) => (
    <button
        onClick={(e) => {
            e.stopPropagation();
            e.preventDefault();
            onClick();
        }}
        onMouseDown={(e) => e.stopPropagation()}
        style={{
            background: 'transparent',
            border: 'none',
            color: danger ? '#ff6666' : '#eee',
            padding: '8px 12px',
            textAlign: 'left',
            cursor: 'pointer',
            borderRadius: '4px',
            fontSize: '14px',
            transition: 'background 0.2s',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            width: '100%',
        }}
        onMouseEnter={(e) => e.currentTarget.style.background = danger ? '#442222' : '#444'}
        onMouseLeave={(e) => e.currentTarget.style.background = 'transparent'}
    >
        <span>{icon}</span>
        <span>{label}</span>
    </button>
);

const Divider = () => (
    <div style={{ height: '1px', background: '#444', margin: '4px 0' }} />
);

const ContextMenu: React.FC<ContextMenuProps> = ({ x, y, targetName, targetInfo, onClose, onAction }) => {
    return (
        <div
            onClick={(e) => e.stopPropagation()}
            onMouseDown={(e) => e.stopPropagation()}
            onPointerDown={(e) => e.stopPropagation()}
            style={{
                position: 'absolute',
                top: y,
                left: x,
                backgroundColor: 'rgba(30, 30, 30, 0.95)',
                border: '1px solid #444',
                borderRadius: '8px',
                padding: '6px',
                zIndex: 2000,
                boxShadow: '0 4px 12px rgba(0,0,0,0.5)',
                minWidth: '180px',
                backdropFilter: 'blur(4px)'
            }}>
            {/* Header with target info */}
            <div style={{
                padding: '8px 10px',
                borderBottom: '1px solid #444',
                marginBottom: '4px',
                fontSize: '13px'
            }}>
                <div style={{ fontWeight: 'bold', color: '#ffcc00', marginBottom: '4px' }}>
                    {targetName}
                </div>
                {targetInfo && (
                    <div style={{ color: '#888', fontSize: '11px' }}>
                        <div>Shape: {targetInfo.shape || 'box'}</div>
                        <div>Size: {targetInfo.size?.join(' × ') || '1 × 1 × 1'}</div>
                        <div>Type: {targetInfo.isNpc ? 'NPC (Active)' : 'Static Object'}</div>
                    </div>
                )}
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
                {/* Behavior Injection */}
                <MenuButton
                    icon="✨"
                    label="Edit Behavior"
                    onClick={() => onAction('injectBehavior')}
                />

                {/* Color Change */}
                <MenuButton
                    icon="🎨"
                    label="Change Color"
                    onClick={() => onAction('changeColor')}
                />

                {/* Size Change */}
                <MenuButton
                    icon="📐"
                    label="Change Size"
                    onClick={() => onAction('changeSize')}
                />

                {/* Shader Selection */}
                <div style={{ position: 'relative' }}>
                    <div style={{
                        color: '#aaa', fontSize: 11, padding: '4px 12px', marginTop: 4
                    }}>
                        Shader
                    </div>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, padding: '0 8px' }}>
                        {['none', 'hologram', 'dissolve', 'pulse', 'rainbow', 'noise'].map((shader) => (
                            <button
                                key={shader}
                                onClick={(e) => { e.stopPropagation(); onAction('changeShader', shader); }}
                                style={{
                                    background: '#333', border: '1px solid #555', color: '#eee',
                                    padding: '4px 8px', borderRadius: 4, fontSize: 11, cursor: 'pointer'
                                }}
                                onMouseEnter={(e) => e.currentTarget.style.background = '#555'}
                                onMouseLeave={(e) => e.currentTarget.style.background = '#333'}
                            >
                                {shader}
                            </button>
                        ))}
                    </div>
                </div>

                {/* Duplicate */}
                <MenuButton
                    icon="📋"
                    label="Duplicate"
                    onClick={() => onAction('duplicate')}
                />

                <Divider />

                {/* Delete */}
                <MenuButton
                    icon="🗑️"
                    label="Delete"
                    danger
                    onClick={() => onAction('delete')}
                />

                <Divider />

                {/* Cancel */}
                <button
                    onClick={onClose}
                    style={{
                        background: 'transparent',
                        border: 'none',
                        color: '#888',
                        padding: '6px 12px',
                        textAlign: 'center',
                        cursor: 'pointer',
                        borderRadius: '4px',
                        fontSize: '12px',
                        width: '100%',
                    }}
                    onMouseEnter={(e) => e.currentTarget.style.background = '#333'}
                    onMouseLeave={(e) => e.currentTarget.style.background = 'transparent'}
                >
                    Cancel
                </button>
            </div>
        </div>
    );
};

export default ContextMenu;
