import std/strutils
import crown/core

proc escHtml(t: string): string =
  t.multiReplace(@[
    ("&", "&amp;"), ("<", "&lt;"), (">", "&gt;"),
    ("\"", "&quot;"), ("'", "&#39;")
  ])

proc isValidEmail(s: string): bool =
  let t = s.strip()
  if t.len < 5 or t.len > 254:
    return false
  let at = t.find('@')
  if at < 1 or at == t.high:
    return false
  let dot = t.find('.', start = at + 1)
  return dot > at + 1 and dot < t.high

proc post*(req: Request): string =
  let email = req.postParams.getOrDefault("email", "").strip()
  if not isValidEmail(email):
    return html"""
      <div class="rounded-xl border border-rose-500/30 bg-rose-950/40 px-4 py-3 text-sm text-rose-100">
        Please enter a valid email address.
      </div>
    """
  return html"""
    <div class="rounded-xl border border-lime-300/35 bg-emerald-950/45 px-4 py-3 text-sm text-lime-50">
      You are on the private preview list. We will contact <span class="font-mono text-lime-200">{escHtml(email)}</span> when Elfentier is ready.
    </div>
  """

proc page*(req: Request): string =
  discard req
  return html"""
    <article class="overflow-visible">
      <section class="relative grid gap-8 px-5 py-8 sm:px-8 lg:min-h-[720px] lg:grid-cols-[0.95fr_1.05fr] lg:gap-4 lg:px-12 lg:py-12">
        <div class="absolute inset-x-0 top-20 h-px overflow-hidden opacity-80">
          <div class="scanline h-px w-full"></div>
        </div>

        <div class="relative z-10 order-2 flex flex-col gap-3 lg:order-1">
          <div class="space-y-6">
            <div class="reveal-up inline-flex items-center gap-3 rounded-full border border-lime-100/15 bg-emerald-950/35 px-3 py-2 text-xs font-semibold uppercase tracking-[0.24em] text-lime-100/80">
              <span class="h-2 w-2 rounded-full bg-lime-300 shadow-[0_0_24px_rgba(190,242,100,0.95)]"></span>
              Coming into signal
            </div>

            <div class="space-y-6">
              <h1 class="reveal-up delay-1 font-display text-4xl font-extrabold leading-[0.96] tracking-[-0.05em] text-white sm:text-6xl lg:text-7xl lg:leading-[0.94] lg:tracking-[-0.055em]">
                Give your AI natural reactions.
              </h1>
              <p class="reveal-up delay-2 max-w-2xl text-base leading-7 text-lime-50/70 sm:text-xl sm:leading-8">
                Elfentier is a lightweight platform for local AI agents to express reflexive, non-verbal responses through motion, sound, and presence.
              </p>
            </div>

            <form
              class="reveal-up delay-3 grid gap-3 rounded-2xl border border-lime-100/15 bg-emerald-950/35 p-2 shadow-2xl shadow-lime-950/20 sm:grid-cols-[1fr_auto]"
              crown-post="/"
              crown-target="#waitlist-status"
            >
              <label class="sr-only" for="email">Email address</label>
              <input
                id="email"
                type="email"
                name="email"
                required
                autocomplete="email"
                placeholder="you@example.com"
                class="min-h-12 min-w-0 rounded-xl border border-transparent bg-lime-50/[0.08] px-4 text-white placeholder:text-lime-100/35 outline-none transition focus:border-lime-300/50 focus:bg-lime-50/[0.12] focus:ring-2 focus:ring-lime-300/25"
              />
              <button
                type="submit"
                class="min-h-12 cursor-pointer rounded-xl bg-lime-200 px-6 text-sm font-bold uppercase tracking-[0.16em] text-emerald-950 shadow-lg shadow-lime-300/15 transition hover:bg-lime-100 focus:outline-none focus:ring-2 focus:ring-lime-300/70 active:scale-[0.98]"
              >
                Request access
              </button>
            </form>
            <div id="waitlist-status" class="min-h-[0.5rem] text-left" aria-live="polite"></div>
          </div>

          <div class="grid gap-3 text-left sm:grid-cols-3">
            <div class="rounded-2xl border border-lime-100/15 bg-lime-50/[0.05] p-4">
              <p class="text-[0.65rem] font-semibold uppercase tracking-[0.22em] text-lime-200/85">Reflex</p>
              <p class="mt-3 text-sm leading-6 text-lime-50/55">Instant, non-verbal reactions before a full thought forms.</p>
            </div>
            <div class="rounded-2xl border border-lime-100/15 bg-lime-50/[0.05] p-4">
              <p class="text-[0.65rem] font-semibold uppercase tracking-[0.22em] text-lime-200/85">Local</p>
              <p class="mt-3 text-sm leading-6 text-lime-50/55">Designed around agents and models that run close to you.</p>
            </div>
            <div class="rounded-2xl border border-lime-100/15 bg-lime-50/[0.05] p-4">
              <p class="text-[0.65rem] font-semibold uppercase tracking-[0.22em] text-lime-200/85">Presence</p>
              <p class="mt-3 text-sm leading-6 text-lime-50/55">Motion, audio, and subtle signals make AI feel less static.</p>
            </div>
          </div>
        </div>

        <div class="relative order-1 min-h-[380px] overflow-visible sm:min-h-[480px] lg:order-2 lg:min-h-0">
          <div class="absolute left-1/2 top-0 z-30 -translate-x-1/2 text-center sm:top-[4%] lg:left-[55%] lg:top-[14%]">
            <p class="font-display text-4xl font-extrabold leading-none tracking-[-0.07em] text-white sm:text-6xl lg:text-8xl">Elfentier</p>
            <p class="mt-2 whitespace-nowrap text-[0.62rem] font-semibold uppercase tracking-[0.22em] text-lime-100/75 sm:mt-3 sm:text-xs sm:tracking-[0.32em] lg:text-sm">Natural reactions for AI agents</p>
          </div>

          <div class="pulse-orbit absolute left-1/2 top-[54%] h-[24rem] w-[24rem] -translate-x-1/2 -translate-y-1/2 rounded-full bg-[radial-gradient(circle,rgba(132,204,22,0.20),rgba(34,197,94,0.08)_38%,transparent_64%)] sm:h-[32rem] sm:w-[32rem] lg:left-[55%] lg:top-[49%] lg:h-[38rem] lg:w-[38rem]"></div>
          <div class="absolute left-1/2 top-[54%] h-[17rem] w-[17rem] -translate-x-1/2 -translate-y-1/2 rounded-full bg-[radial-gradient(circle,transparent_58%,rgba(190,242,100,0.055)_59%,transparent_70%)] sm:h-[22rem] sm:w-[22rem] lg:left-[55%] lg:top-[49%] lg:h-[26rem] lg:w-[26rem]"></div>
          <div class="absolute left-[22%] top-[31%] h-2 w-2 rounded-full bg-lime-200 shadow-[0_0_28px_rgba(190,242,100,0.95)]"></div>
          <div class="absolute right-[18%] top-[36%] h-1.5 w-1.5 rounded-full bg-emerald-200/90 shadow-[0_0_22px_rgba(167,243,208,0.8)]"></div>
          <div class="absolute bottom-[16%] left-[28%] h-1.5 w-1.5 rounded-full bg-white/80 shadow-[0_0_18px_rgba(255,255,255,0.75)]"></div>

          <svg
            class="wing-drift crow-silhouette absolute right-[-4rem] top-[20%] z-10 w-[30rem] max-w-[128vw] -translate-y-1/2 drop-shadow-[0_28px_70px_rgba(0,0,0,0.82)] sm:right-[-1rem] sm:top-[20%] sm:w-[36rem] lg:right-[-2rem] lg:top-[25%] lg:w-[42rem] lg:max-w-[132vw] lg:drop-shadow-[0_36px_90px_rgba(0,0,0,0.82)]"
            viewBox="0 0 900 620"
            role="img"
            aria-label="Crow silhouette"
            xmlns="http://www.w3.org/2000/svg"
          >
            <defs>
              <linearGradient id="halo" x1="90" y1="48" x2="752" y2="572" gradientUnits="userSpaceOnUse">
                <stop stop-color="#BEF264" stop-opacity="0.58"/>
                <stop offset="0.55" stop-color="#22C55E" stop-opacity="0.22"/>
                <stop offset="1" stop-color="#052E16" stop-opacity="0"/>
              </linearGradient>
              <linearGradient id="signal" x1="200" y1="370" x2="690" y2="270" gradientUnits="userSpaceOnUse">
                <stop stop-color="#14532D" stop-opacity="0"/>
                <stop offset="0.32" stop-color="#BEF264" stop-opacity="0.78"/>
                <stop offset="0.66" stop-color="#86EFAC" stop-opacity="0.92"/>
                <stop offset="1" stop-color="#22C55E" stop-opacity="0"/>
              </linearGradient>
              <filter id="softGlow" x="-20%" y="-20%" width="140%" height="140%">
                <feGaussianBlur stdDeviation="5" result="blur"/>
                <feMerge>
                  <feMergeNode in="blur"/>
                  <feMergeNode in="SourceGraphic"/>
                </feMerge>
              </filter>
              <clipPath id="crowClip">
                <path d="M274 356C314 264 417 203 536 207C610 210 666 242 704 296C677 376 603 438 503 459C395 482 316 438 274 356Z"/>
                <path d="M526 217C560 176 626 162 682 191C720 194 760 210 786 236C741 239 707 249 678 267C632 245 584 249 546 281C545 255 538 235 526 217Z"/>
                <path d="M280 369L80 500C116 452 174 396 266 337Z"/>
                <path d="M266 386L112 547C184 520 256 476 335 414Z"/>
                <path d="M329 329C250 272 170 239 73 220C129 293 196 367 274 444C282 395 299 357 329 329Z"/>
                <path d="M324 353C300 418 288 485 289 560C348 510 402 454 455 391C408 379 366 366 324 353Z"/>
              </clipPath>
            </defs>
            <path d="M116 466C212 309 363 204 548 207C656 209 748 242 821 308C750 294 699 304 650 343C564 413 463 467 339 468C252 469 179 461 116 466Z" fill="url(#halo)" opacity="0.68"/>
            <path d="M280 369L80 500C116 452 174 396 266 337Z" fill="#020409"/>
            <path d="M266 386L112 547C184 520 256 476 335 414Z" fill="#020409"/>
            <path d="M329 329C250 272 170 239 73 220C129 293 196 367 274 444C282 395 299 357 329 329Z" fill="#020409"/>
            <path d="M274 356C314 264 417 203 536 207C610 210 666 242 704 296C677 376 603 438 503 459C395 482 316 438 274 356Z" fill="#020409"/>
            <path d="M526 217C560 176 626 162 682 191C720 194 760 210 786 236C741 239 707 249 678 267C632 245 584 249 546 281C545 255 538 235 526 217Z" fill="#020409"/>
            <path d="M676 196C720 193 766 208 802 237L704 239C697 221 688 207 676 196Z" fill="#020409"/>
            <path d="M324 353C300 418 288 485 289 560C348 510 402 454 455 391C408 379 366 366 324 353Z" fill="#020409"/>
            <path d="M360 349C310 331 253 326 182 338C239 363 295 398 351 443C346 404 349 373 360 349Z" fill="#05070D"/>
            <path d="M382 329C323 274 257 235 177 214C221 283 274 347 337 405C346 372 361 347 382 329Z" fill="#05070D"/>
            <path d="M433 438C443 470 456 498 475 526" stroke="#020409" stroke-width="13" stroke-linecap="round" fill="none"/>
            <path d="M505 432C515 468 532 494 552 518" stroke="#020409" stroke-width="13" stroke-linecap="round" fill="none"/>
            <path d="M474 526C445 531 422 542 404 560M474 526C499 533 520 535 545 530M474 526C468 541 462 552 452 563" stroke="#020409" stroke-width="9" stroke-linecap="round" fill="none"/>
            <path d="M552 518C523 532 504 548 492 570M552 518C579 523 603 522 629 511M552 518C552 535 548 548 539 563" stroke="#020409" stroke-width="9" stroke-linecap="round" fill="none"/>
            <path d="M642 239C656 230 678 233 692 249C667 253 647 262 631 279C630 264 634 250 642 239Z" fill="#05070D"/>
            <circle cx="671" cy="226" r="4.2" fill="#BEF264"/>
            <g clip-path="url(#crowClip)" filter="url(#softGlow)">
              <path d="M198 425C292 340 381 333 475 367C548 394 620 378 696 318" stroke="url(#signal)" stroke-width="9" stroke-linecap="round" fill="none" opacity="0.95"/>
              <path d="M242 470C342 398 443 382 556 407" stroke="#BEF264" stroke-opacity="0.42" stroke-width="4" stroke-linecap="round" fill="none"/>
              <path d="M318 366C386 300 470 279 573 306" stroke="#4ADE80" stroke-opacity="0.45" stroke-width="4" stroke-linecap="round" fill="none"/>
              <circle cx="372" cy="344" r="4" fill="#BEF264"/>
              <circle cx="456" cy="382" r="3.5" fill="#86EFAC"/>
              <circle cx="562" cy="350" r="3" fill="#BBF7D0"/>
              <circle cx="618" cy="314" r="2.5" fill="#D9F99D"/>
            </g>
          </svg>
        </div>
      </section>
    </article>
  """
