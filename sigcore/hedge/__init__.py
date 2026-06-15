"""Signature hedging phase.

Turns the validated core primitives into a hedger: given a liability, find the
self-financing trading strategy minimising a chosen risk functional of the
shortfall, under the risk-neutral measure. Single asset; closed-form-covector
payoffs (forward, Asian); mean-variance plus one convex asymmetric penalty.

The enlarged path is time-augmented + Hoff lead-lag of price, with channels
``0 = time``, ``1 = lead``, ``2 = lag`` and trade letter = the lead channel
(see embedding.py). Appending the trade letter to a strategy covector yields the
Ito trading integral — the mechanism validated by core test I.
"""
