"""One canonical name per club, with the aliases used by each data source.

Any name not found raises ``UnmappedTeamError`` so new promoted clubs are caught loudly.
"""

from __future__ import annotations

# canonical name -> extra aliases (the football-data.co.uk spelling is added automatically
# for names that are written differently there).
_CANONICAL: dict[str, list[str]] = {
    "Arsenal": [],
    "Aston Villa": [],
    "Barnsley": [],
    "Birmingham City": ["Birmingham"],
    "Blackburn Rovers": ["Blackburn"],
    "Blackpool": [],
    "Bolton Wanderers": ["Bolton"],
    "AFC Bournemouth": ["Bournemouth"],
    "Brentford": [],
    "Brighton & Hove Albion": ["Brighton", "Brighton and Hove Albion"],
    "Bristol City": [],
    "Burnley": [],
    "Burton Albion": ["Burton"],
    "Cardiff City": ["Cardiff"],
    "Charlton Athletic": ["Charlton"],
    "Chelsea": [],
    "Coventry City": ["Coventry"],
    "Crystal Palace": [],
    "Derby County": ["Derby"],
    "Everton": [],
    "Fulham": [],
    "Huddersfield Town": ["Huddersfield"],
    "Hull City": ["Hull"],
    "Ipswich Town": ["Ipswich"],
    "Leeds United": ["Leeds"],
    "Leicester City": ["Leicester"],
    "Lincoln City": ["Lincoln"],
    "Liverpool": [],
    "Luton Town": ["Luton"],
    "Manchester City": ["Man City"],
    "Manchester United": ["Man United", "Man Utd"],
    "Middlesbrough": [],
    "Millwall": [],
    "Milton Keynes Dons": ["MK Dons"],
    "Newcastle United": ["Newcastle"],
    "Norwich City": ["Norwich"],
    "Nottingham Forest": ["Nott'm Forest", "Nottm Forest", "Nott'ham Forest"],
    "Oxford United": ["Oxford"],
    "Peterborough United": ["Peterboro", "Peterborough"],
    "Plymouth Argyle": ["Plymouth"],
    "Portsmouth": [],
    "Preston North End": ["Preston"],
    "Queens Park Rangers": ["QPR"],
    "Reading": [],
    "Rotherham United": ["Rotherham"],
    "Sheffield United": ["Sheffield Utd"],
    "Sheffield Wednesday": ["Sheffield Weds"],
    "Southampton": [],
    "Stoke City": ["Stoke"],
    "Sunderland": [],
    "Swansea City": ["Swansea"],
    "Tottenham Hotspur": ["Tottenham", "Spurs"],
    "Watford": [],
    "West Bromwich Albion": ["West Brom"],
    "West Ham United": ["West Ham"],
    "Wigan Athletic": ["Wigan"],
    "Wolverhampton Wanderers": ["Wolves", "Wolverhampton"],
    "Wrexham": [],
    "Wycombe Wanderers": ["Wycombe"],
}

# canonical -> short display name for charts
_SHORT: dict[str, str] = {
    "Manchester City": "Man City",
    "Manchester United": "Man Utd",
    "Tottenham Hotspur": "Spurs",
    "Nottingham Forest": "Forest",
    "Newcastle United": "Newcastle",
    "Brighton & Hove Albion": "Brighton",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves",
    "Crystal Palace": "Palace",
    "Leeds United": "Leeds",
    "Leicester City": "Leicester",
    "AFC Bournemouth": "Bournemouth",
}


class UnmappedTeamError(KeyError):
    """Raised when a team name from a source has no canonical mapping."""


def _build_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}
    for canon, aliases in _CANONICAL.items():
        for name in [canon, *aliases]:
            key = name.strip().lower()
            if key in lookup and lookup[key] != canon:
                raise ValueError(f"alias {name!r} maps to two clubs")
            lookup[key] = canon
    return lookup


_LOOKUP = _build_lookup()


def canonical(name: str) -> str:
    """Return the canonical club name or raise ``UnmappedTeamError``."""
    try:
        return _LOOKUP[str(name).strip().lower()]
    except KeyError:
        raise UnmappedTeamError(f"Unmapped team name: {name!r}") from None


def short_name(canon: str) -> str:
    """Short display name for a canonical name."""
    return _SHORT.get(canon, canon)


def all_canonical() -> list[str]:
    return sorted(_CANONICAL)
