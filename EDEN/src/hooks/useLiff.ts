import { useEffect, useState } from 'react';
import liff from '@line/liff';

const LIFF_ID = import.meta.env.VITE_LIFF_ID || '';

export interface LiffProfile {
    userId: string;
    displayName: string;
    pictureUrl?: string;
    statusMessage?: string;
}

export const useLiff = () => {
    const [liffError, setLiffError] = useState<string | null>(null);
    const [profile, setProfile] = useState<LiffProfile | null>(null);
    const [isLoggedIn, setIsLoggedIn] = useState(false);
    const [isInitialized, setIsInitialized] = useState(false);

    useEffect(() => {
        // If no LIFF ID is provided, skip initialization but mark as initialized so the app doesn't hang
        if (!LIFF_ID) {
            console.warn("LIFF ID is not set in environment variables (VITE_LIFF_ID)");
            setIsInitialized(true);
            return;
        }

        liff.init({ liffId: LIFF_ID })
            .then(() => {
                setIsInitialized(true);
                if (liff.isLoggedIn()) {
                    setIsLoggedIn(true);
                    liff.getProfile().then(p => {
                        setProfile({
                            userId: p.userId,
                            displayName: p.displayName,
                            pictureUrl: p.pictureUrl,
                            statusMessage: p.statusMessage
                        });
                    }).catch(e => {
                        console.error('LIFF getProfile Error', e);
                    });
                } else {
                    setIsLoggedIn(false);
                }
            })
            .catch((e: Error) => {
                console.error('LIFF Init Error', e);
                setLiffError(e.message);
                setIsInitialized(true);
            });
    }, []);

    const login = () => {
        if (!LIFF_ID) return;
        if (!liff.isLoggedIn()) {
            liff.login();
        }
    };

    return { liff, liffError, profile, isLoggedIn, isInitialized, login };
};
