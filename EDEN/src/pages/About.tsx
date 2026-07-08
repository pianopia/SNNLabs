import React, { useEffect } from 'react';
import { Link } from 'react-router-dom';

const About: React.FC = () => {
    const featureCards = [
        {
            title: '3層アーキテクチャ',
            description: 'ブラウザ描画、リアルタイム同期サーバー、Map/AIサービスを分離。生成・実行・表示を役割分担し、安定した運用を実現します。'
        },
        {
            title: 'リアルタイム同期',
            description: 'WebSocketベースでプレイヤー移動・チャット・状態変化を即時共有。複数ユーザーが同じ世界を同じタイミングで体験できます。'
        },
        {
            title: 'AIによるエンティティ生成',
            description: 'Gemini 3.0 Proで、自然言語からオブジェクト/NPCのプロパティと挙動コードをJSONで生成。保存後すぐに世界へ反映します。'
        },
        {
            title: 'サンドボックス実行',
            description: 'Node.js VMを使い、NPCごとに分離されたコンテキストでスクリプトを実行。エラー時も全体停止を避ける設計です。'
        },
        {
            title: '永続ワールド',
            description: 'SQLite + Drizzle ORMでエンティティとチャンクを管理。再接続後も世界の状態が維持される、継続的な空間を提供します。'
        },
        {
            title: '描画最適化',
            description: 'React Three Fiber上でLODとFloating Originを採用。無限に広がる空間でも描画精度とパフォーマンスを両立します。'
        }
    ];

    const servicePoints = [
        {
            title: '体験',
            text: 'プレイヤー同士が移動・会話しながら、AIが生成したオブジェクトやNPCが動き続ける「生きた3D空間」を提供します。'
        },
        {
            title: '技術基盤',
            text: 'React Three Fiber / Three.js、Node.js WebSocket、SQLite、Gemini 3.0 Proを統合し、MVP段階で必要な同期・生成・永続化を実装済みです。'
        },
        {
            title: '信頼性',
            text: '動的コード実行はサンドボックスで隔離し、クライアント・サーバー双方で整合性を担保。実運用を見据えた堅牢性を重視しています。'
        }
    ];

    useEffect(() => {
        document.title = 'About - EDEN14 | AIリアルタイム生成メタバース';

        // Update meta description
        const metaDescription = document.querySelector('meta[name="description"]');
        if (metaDescription) {
            metaDescription.setAttribute('content', 'EDEN14の深層へ。Gemini 3.0 ProとNode.js VM、React Three Fiberが描く自律生成エコシステムの静かなる鼓動と、その技術仕様について。');
        }

        return () => {
            document.title = 'EDEN14 - AIリアルタイム生成メタバース';
            if (metaDescription) {
                // Restore default description
                metaDescription.setAttribute('content', 'EDEN14は、Gemini 3.0 Proとコードが紡ぐ『自己進化型AIメタバース』。言葉ひとつで物理法則が書き換わり、静寂の中で世界が無限に生成されていく。ブラウザの向こう側に広がる、終わりのないデジタルパラダイス。');
            }
        };
    }, []);
    return (
        <>
            <div style={{
                minHeight: '100vh',
                background: 'linear-gradient(180deg, #000 0%, #0a1a25 50%, #000 100%)',
                color: 'white',
                fontFamily: "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Ubuntu, 'Helvetica Neue', sans-serif",
                overflowX: 'hidden',
                width: '100%'
            }}>
                {/* Header */}
                <header style={{
                    position: 'fixed',
                    top: 0,
                    left: 0,
                    right: 0,
                    padding: '20px 5%',
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    background: 'rgba(0,0,0,0.5)',
                    backdropFilter: 'blur(10px)',
                    zIndex: 1000
                }}>
                    <Link to="/" style={{ textDecoration: 'none' }}>
                        <img src="/eden_logo_text.svg" alt="EDEN" style={{ height: '40px' }} />
                    </Link>
                    <nav style={{ display: 'flex', gap: '30px' }}>
                        <Link to="/" style={{ color: '#aaa', textDecoration: 'none', fontSize: '14px', letterSpacing: '1px' }}>HOME</Link>
                        <Link to="/about" style={{ color: '#00ffff', textDecoration: 'none', fontSize: '14px', letterSpacing: '1px' }}>ABOUT</Link>
                    </nav>
                </header>

                {/* Hero */}
                <section style={{
                    minHeight: '100vh',
                    display: 'flex',
                    flexDirection: 'column',
                    justifyContent: 'center',
                    alignItems: 'center',
                    textAlign: 'center',
                    padding: '120px 5% 80px',
                    position: 'relative',
                    width: '100%',
                    boxSizing: 'border-box'
                }}>
                    <div style={{
                        position: 'absolute',
                        width: '80vw',
                        height: '80vw',
                        maxWidth: '800px',
                        maxHeight: '800px',
                        background: 'radial-gradient(circle, rgba(0,255,255,0.1) 0%, transparent 70%)',
                        borderRadius: '50%',
                        filter: 'blur(60px)',
                        zIndex: 0,
                        top: '50%',
                        left: '50%',
                        transform: 'translate(-50%, -50%)'
                    }} />

                    <h1 style={{
                        fontSize: 'clamp(3rem, 10vw, 6rem)',
                        fontWeight: '300',
                        letterSpacing: '0.2em',
                        margin: 0,
                        background: 'linear-gradient(90deg, #00ffff, #00ff88)',
                        WebkitBackgroundClip: 'text',
                        WebkitTextFillColor: 'transparent',
                        zIndex: 1,
                        width: '100%'
                    }}>
                        EDEN14
                    </h1>

                    <p style={{
                        fontSize: 'clamp(1rem, 2.5vw, 1.5rem)',
                        color: '#888',
                        letterSpacing: '0.3em',
                        marginTop: '20px',
                        textTransform: 'uppercase',
                        zIndex: 1
                    }}>
                        Digital Nature World
                    </p>

                    <p style={{
                        fontSize: 'clamp(1rem, 1.5vw, 1.2rem)',
                        color: '#aaa',
                        maxWidth: '800px',
                        width: '100%',
                        lineHeight: '1.8',
                        marginTop: '40px',
                        zIndex: 1
                    }}>
                        EDEN14は、AIによってオブジェクト・挙動・空間がリアルタイムに生成され続けるメタバース基盤です。<br />
                        「同期された体験」と「安全な動的実行」を両立し、創造と交流が同時に進化する世界を目指しています。
                    </p>
                </section>

                {/* Service Introduction */}
                <section style={{
                    padding: '0 5% 80px',
                    width: '100%',
                    boxSizing: 'border-box'
                }}>
                    <div style={{
                        maxWidth: '1200px',
                        margin: '0 auto',
                        background: 'linear-gradient(135deg, rgba(0,255,255,0.08), rgba(0,255,136,0.04))',
                        border: '1px solid rgba(0,255,255,0.2)',
                        borderRadius: '24px',
                        padding: '40px'
                    }}>
                        <h2 style={{
                            fontSize: 'clamp(1.8rem, 3vw, 2.4rem)',
                            fontWeight: '300',
                            margin: '0 0 20px 0',
                            letterSpacing: '0.08em'
                        }}>
                            Service Overview
                        </h2>
                        <p style={{
                            color: '#9bd7d7',
                            lineHeight: '1.9',
                            margin: '0 0 28px 0'
                        }}>
                            クライアント（描画・予測）/ リアルタイムサーバー（同期・ロジック）/ Map・AIサービス（生成・永続化）の3層で構成。
                            ユーザー入力や未踏エリアへの到達をトリガーにAI生成を行い、データベース保存と同時にワールドへ即時デプロイします。
                        </p>

                        <div style={{
                            width: '100%',
                            display: 'flex',
                            justifyContent: 'center',
                            marginBottom: '40px'
                        }}>
                            <img
                                src="/service_capture.png"
                                alt="Service Capture"
                                style={{
                                    width: '100%',
                                    maxWidth: '800px',
                                    borderRadius: '12px',
                                    boxShadow: '0 8px 32px rgba(0, 255, 255, 0.15)',
                                    border: '1px solid rgba(0, 255, 255, 0.2)'
                                }}
                            />
                        </div>

                        <div style={{
                            display: 'grid',
                            gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
                            gap: '20px'
                        }}>
                            {servicePoints.map((point) => (
                                <div key={point.title} style={{
                                    background: 'rgba(0,0,0,0.25)',
                                    border: '1px solid rgba(255,255,255,0.08)',
                                    borderRadius: '16px',
                                    padding: '20px'
                                }}>
                                    <h3 style={{
                                        margin: '0 0 10px 0',
                                        color: '#00ffff',
                                        fontSize: '1rem',
                                        letterSpacing: '0.05em'
                                    }}>
                                        {point.title}
                                    </h3>
                                    <p style={{
                                        margin: 0,
                                        color: '#9aa4ad',
                                        lineHeight: '1.7',
                                        fontSize: '0.95rem'
                                    }}>
                                        {point.text}
                                    </p>
                                </div>
                            ))}
                        </div>
                    </div>
                </section>

                {/* Features */}
                <section style={{
                    padding: '100px 5%',
                    margin: '0 auto',
                    width: '100%',
                    boxSizing: 'border-box'
                }}>
                    <h2 style={{
                        fontSize: 'clamp(2rem, 4vw, 3rem)',
                        fontWeight: '300',
                        textAlign: 'center',
                        marginBottom: '80px',
                        color: '#fff',
                        letterSpacing: '0.1em'
                    }}>
                        Features
                    </h2>

                    <div style={{
                        display: 'grid',
                        gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))',
                        gap: '40px',
                        maxWidth: '1400px',
                        margin: '0 auto'
                    }}>
                        {featureCards.map((feature, index) => (
                            <div key={index} style={{
                                background: 'rgba(255,255,255,0.03)',
                                border: '1px solid rgba(255,255,255,0.08)',
                                borderRadius: '24px',
                                padding: '40px',
                                transition: 'all 0.3s',
                                display: 'flex',
                                flexDirection: 'column',
                                alignItems: 'flex-start',
                                minHeight: '160px'
                            }}>
                                <h3 style={{
                                    fontSize: '1.4rem',
                                    fontWeight: '500',
                                    color: '#00ffff',
                                    marginBottom: '20px',
                                    marginTop: 0
                                }}>
                                    {feature.title}
                                </h3>
                                <p style={{
                                    fontSize: '1rem',
                                    color: '#888',
                                    lineHeight: '1.8',
                                    margin: 0
                                }}>
                                    {feature.description}
                                </p>
                            </div>
                        ))}
                    </div>
                </section>

                {/* CTA */}
                <section style={{
                    padding: '100px 5%',
                    textAlign: 'center'
                }}>
                    <h2 style={{
                        fontSize: 'clamp(2rem, 4vw, 3rem)',
                        fontWeight: '300',
                        marginBottom: '40px',
                        color: '#fff'
                    }}>
                        Ready to Explore?
                    </h2>

                    <Link to="/" style={{
                        display: 'inline-block',
                        padding: '20px 60px',
                        background: 'linear-gradient(90deg, #00ffff, #00ff88)',
                        color: '#000',
                        textDecoration: 'none',
                        borderRadius: '16px',
                        fontWeight: 'bold',
                        fontSize: '1.2rem',
                        letterSpacing: '1px',
                        textTransform: 'uppercase',
                        boxShadow: '0 0 40px rgba(0,255,255,0.3)',
                        transition: 'all 0.3s'
                    }}>
                        Enter EDEN14
                    </Link>
                </section>

                {/* Footer */}
                <footer style={{
                    padding: '40px 5%',
                    borderTop: '1px solid rgba(255,255,255,0.05)',
                    textAlign: 'center'
                }}>
                    <p style={{ color: '#555', fontSize: '0.9rem', margin: 0 }}>
                        © 2026 EDEN14. All rights reserved.
                    </p>
                </footer>
            </div>
        </>
    );
};

export default About;
