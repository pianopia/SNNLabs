import crown/core

proc layout*(content: string): string =
  return html"""
<!DOCTYPE html>
<html lang="en" class="h-full">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Elfentier — Natural reactions for AI agents</title>
  <meta name="description" content="Give your AI natural reactions. Elfentier is a lightweight platform for expressive local AI agents.">
  <link rel="icon" href="/favicon.svg" type="image/svg+xml">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Work+Sans:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {{
      --font-display: "Outfit", ui-sans-serif, system-ui, sans-serif;
      --font-body: "Work Sans", ui-sans-serif, system-ui, sans-serif;
      --ink: #041108;
      --surface: #08170d;
      --line: rgba(187, 247, 208, 0.18);
      --glow: rgba(132, 204, 22, 0.26);
    }}
    body {{ font-family: var(--font-body); }}
    .font-display {{ font-family: var(--font-display); }}
    .noise {{
      background-image:
        radial-gradient(circle at 20% 20%, rgba(236,252,203,0.10) 0 1px, transparent 1px),
        radial-gradient(circle at 82% 2%, rgba(132,204,22,0.20), transparent 28rem),
        radial-gradient(circle at 18% 82%, rgba(34,197,94,0.18), transparent 24rem),
        radial-gradient(circle at 50% 120%, rgba(190,242,100,0.12), transparent 32rem);
      background-size: 14px 14px, auto, auto;
    }}
    .grid-fade {{
      background-image:
        linear-gradient(to right, rgba(187,247,208,0.09) 1px, transparent 1px),
        linear-gradient(to bottom, rgba(187,247,208,0.09) 1px, transparent 1px);
      background-size: 56px 56px;
      mask-image: radial-gradient(ellipse 70% 55% at 50% 42%, black 0%, transparent 76%);
    }}
    .scanline {{
      background: linear-gradient(90deg, transparent, rgba(190,242,100,0.36), transparent);
      animation: sweep 7s ease-in-out infinite;
    }}
    .wing-drift {{ animation: wing-drift 8s ease-in-out infinite; }}
    .pulse-orbit {{ animation: pulse-orbit 5s ease-in-out infinite; }}
    .reveal-up {{ animation: reveal-up 760ms cubic-bezier(.2,.8,.2,1) both; }}
    .delay-1 {{ animation-delay: 90ms; }}
    .delay-2 {{ animation-delay: 180ms; }}
    .delay-3 {{ animation-delay: 270ms; }}
    @keyframes sweep {{
      0%, 100% {{ transform: translateX(-45%) scaleX(0.55); opacity: 0; }}
      35%, 65% {{ opacity: 1; }}
      50% {{ transform: translateX(45%) scaleX(1); }}
    }}
    @keyframes wing-drift {{
      0%, 100% {{ transform: translate3d(0,0,0) rotate(-1deg); }}
      50% {{ transform: translate3d(0,-12px,0) rotate(1deg); }}
    }}
    @keyframes pulse-orbit {{
      0%, 100% {{ opacity: .38; transform: scale(.985); }}
      50% {{ opacity: .88; transform: scale(1.015); }}
    }}
    @keyframes reveal-up {{
      from {{ opacity: 0; transform: translateY(18px); }}
      to {{ opacity: 1; transform: translateY(0); }}
    }}
    @media (prefers-reduced-motion: reduce) {{
      *, *::before, *::after {{
        animation-duration: 1ms !important;
        animation-iteration-count: 1 !important;
        scroll-behavior: auto !important;
      }}
    }}
  </style>
</head>
<body class="min-h-full overflow-x-hidden antialiased text-lime-50 bg-[#041108] selection:bg-lime-300/35 selection:text-lime-950">
  <div class="fixed inset-0 -z-10 overflow-hidden">
    <div class="noise absolute inset-0 opacity-80"></div>
    <div class="grid-fade absolute inset-0 opacity-70"></div>
    <div class="absolute left-1/2 top-[-18rem] h-[42rem] w-[42rem] -translate-x-1/2 rounded-full bg-lime-300/16 blur-[130px]"></div>
    <div class="absolute bottom-[-12rem] right-[-10rem] h-[34rem] w-[34rem] rounded-full bg-emerald-400/16 blur-[110px]"></div>
  </div>
  <div class="relative flex min-h-full flex-col">
    <main class="flex-1 px-4 py-6 sm:px-6 sm:py-10">
      <div class="mx-auto max-w-7xl">
        {content}
      </div>
    </main>
    <footer class="px-4 pb-8 text-center text-xs text-lime-100/45 sm:px-6">
      <p>© Elfentier. A quiet signal before the first flight.</p>
    </footer>
  </div>
</body>
</html>
"""
