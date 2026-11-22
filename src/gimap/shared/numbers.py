"""Display numbers in reports and result widgets. Keep the original numbers for calculations and exports."""

from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation, localcontext


def _finite(value):
    if value is None:
        return None
    try:
        number = Decimal(str(value))
    except InvalidOperation:
        return None
    return number if number.is_finite() else None


def _exponent(number, error, place):
    """Use scientific notation for extreme magnitudes and for rounding to tens, hundreds, etc."""
    magnitude = (number or error).adjusted()
    return magnitude if magnitude < -4 or magnitude >= 7 or place > 0 else 0


def _text(number, exponent, suffix=""):
    mantissa = format(number.scaleb(-exponent), "f")
    return f"{mantissa}{suffix}e{exponent:+d}" if exponent else mantissa + suffix


def _significant(number, digits):
    if not number:
        return "0"
    with localcontext() as ctx:
        ctx.prec = digits
        ctx.rounding = ROUND_HALF_EVEN
        rounded = number.normalize()
        exponent = rounded.adjusted()
        if -4 <= exponent < digits:
            return format(rounded, "f")
        return f"{rounded.scaleb(-exponent):f}e{exponent:+03d}"


def format_number(value, error=None, *, compact=False, value_only=False, digits=3) -> str:
    """Round a positive uncertainty to two significant digits, then the value to the same place.

    Decimal ties round to even. In compact notation the parenthesized digits count units of the
    last shown place. With value_only, repeated values and separate columns keep the pair's precision.
    Missing or invalid uncertainties use the ordinary significant-digit display. Zero is explicit.
    """
    number = _finite(value)
    if number is None:
        return "—"
    sigma = _finite(error)
    if sigma is None or sigma <= 0:
        shown = _significant(number, digits)
        if sigma == 0 and not value_only:
            # Insert (0) before an exponent so that it refers to the mantissa's last place.
            mantissa, sep, exponent = shown.partition("e")
            return f"{mantissa}(0){sep}{exponent}" if compact else f"{shown} ± 0"
        return shown

    with localcontext() as ctx:
        place = sigma.adjusted() - 1
        # quantize needs enough coefficient digits even when the two magnitudes are far apart.
        ctx.prec = max(3, number.adjusted() - place + 2)
        quantum = Decimal(1).scaleb(place)
        sigma = sigma.quantize(quantum, rounding=ROUND_HALF_EVEN)
        if sigma.adjusted() - 1 > place:  # e.g. 0.0999 rounds to 0.10
            place += 1
            quantum = Decimal(1).scaleb(place)
            sigma = sigma.quantize(quantum)
        rounded = number.quantize(quantum, rounding=ROUND_HALF_EVEN)
        if not rounded:
            rounded = rounded.copy_abs()
        exponent = _exponent(rounded, sigma, place)
        if value_only:
            return _text(rounded, exponent)
        if compact:
            return _text(rounded, exponent, f"({sigma.scaleb(-place):f})")
        return f"{_text(rounded, exponent)} ± {_text(sigma, _exponent(sigma, sigma, place))}"


def applied_angle(value, measurement=None):
    """Print an applied αi or ψ with the precision of the observation that supplied it."""
    record = measurement or {}
    measured, error = record.get("deg"), record.get("error_deg")
    if "decision" in record:  # αi can retain the nominal angle even when it equals the observation.
        used = record["decision"] in ("disagreement", "measured")
    elif record.get("origin") == "automatic":
        used = record.get("significant", False)
    elif record.get("origin") == "measured":  # Apply from the GUI can select a sub-threshold roll.
        used = True
    else:
        used = not record.get("origin") and value == measured  # older profiles have no origin
    if used and measured is not None and error is not None:
        # Some saved geometries already rounded an angle. A stale, different value must
        # not borrow its precision from the old measurement.
        shown = format_number(value, error, value_only=True)
        if shown == format_number(measured, error, value_only=True):
            return shown
    return format_number(value)
