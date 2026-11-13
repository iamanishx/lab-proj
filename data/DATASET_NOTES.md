# Scam Encounters India - Dataset Notes

Generated 13 Nov 2026 (IST). Companion to scam_encounters_india.csv.

## Size
- 2000 rows, one row per scam attempt/encounter.
- 810 distinct source URLs. Rows sharing a URL are distinct incidents reported in the same complaint thread (original post + commenters), each with its own details.

## Schema (exact column order)
channel (call/sms/whatsapp/email), claimed_sender (bank_kyc/refund/prize/job/family_emergency/other), asked_for (otp/pin/link/qr_scan/money/nothing), urgency_pressure (yes/no), person_daily_upi_use (unknown for every scraped row), engaged (ignored/replied/clicked/shared_otp/lost_money), source_url.

## Class balance
- channel: call 947 (47.3%), email 514 (25.7%), whatsapp 421 (21.1%), sms 118 (5.9%)
- claimed_sender: other 747, prize 722 (36.1%), job 260, bank_kyc 151, refund 81, family_emergency 39
- asked_for: money 1117, nothing 565, link 108, otp 103, qr_scan 68, pin 39
- urgency_pressure: no 1619, yes 381
- engaged: ignored 692, replied 665, lost_money 596, clicked 27, shared_otp 20

## Sources (all individually published public reports, each row traceable via source_url)
- ConsumerComplaints.in individual complaint pages and their comment threads (1229 rows): first-person scam reports across KBC/lottery, fake bank/KYC calls, OLX/army QR scams, data-entry job scams, gigolo/escort job scams, donation calls, challan/legal-threat SMS, fake bank SMS, loan-app extortion, digital arrest, courier/customs, insurance bonus/refund calls.
- Old-style ConsumerComplaints megathreads (KBC lottery, TCS/visa) rule-coded per comment: these carry ~10-15% residual noise and skew prize/lottery.
- Reddit (r/india, r/legaladviceindia, r/scams and similar) first-person posts, hand-coded: high confidence.

## Coding conventions
- Telegram/Instagram/Snapchat-origin messaging scams are coded whatsapp (schema has no telegram); flagged here as a known approximation.
- engaged = most severe outcome wins; asked_for priority otp > qr_scan > pin > link > money > nothing.
- bank_kyc covers bank/wallet/KYC/card impersonation; card/identity-detail asks coded as pin; sharing identity documents (Aadhaar) coded engaged=shared_otp.
- urgency_pressure=yes only on an explicit threat or deadline (legal action, police, expiry, account freeze).
- Marketplace-originated scams (OLX/99acres/magicbricks) coded by the actual contact channel.
- Multiple incidents in one thread URL are separate rows; bare "same thing happened to me" comments without detail are excluded; one incident reposted in two threads counts once.
- Excluded: recovery-scam ads, fake "customer care" replies by scammers, generic warnings/PSAs with no first-person encounter, pure commercial disputes (undelivered products) without a scam approach, non-Indian reporters, in-person-only frauds with no call/sms/whatsapp/email channel.
- person_daily_upi_use is "unknown" for every row: scraped reports do not state the victim's daily UPI use. This column will need labels from the survey/interview data.

## Known caveats
- Prize is still the largest claimed_sender class (~36%) because of the KBC/lottery megathreads; the share keeps falling as new hand-coded sources are added (was ~41% at 1,750 rows).
- Rule-coded megathread rows carry ~10-15% residual noise; hand-coded rows are high confidence.
- Self-reported public complaints skew toward engaged victims (replied/lost_money) relative to the general population.
