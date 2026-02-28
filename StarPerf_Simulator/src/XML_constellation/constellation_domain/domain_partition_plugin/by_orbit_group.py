"""
Domain partition plugin: by_orbit_group

Partition a shell into domains by grouping consecutive orbital planes.
Currently supports Walker-Delta constellations only (inclination <= 80°).

For Walker-Delta (e.g. Starlink 53°):
  - All orbital planes form a ring (first and last planes connected via inter-orbit ISL)
  - Domains also form a ring topology: domain 0 ↔ 1 ↔ ... ↔ (n-1) ↔ 0

For Walker-Star (e.g. OneWeb 87.9°):
  - First and last planes have NO inter-orbit ISL (polar seam)
  - Reserved for future plugin: by_orbit_group_polar.py
"""

from dataclasses import dataclass, field


@dataclass
class DomainInfo:
    """Domain partition result container."""
    n_domains: int
    domain_of_sat: dict  # {sat_id: domain_id}
    domain_sats: dict    # {domain_id: [sat_ids]}
    domain_orbits: dict  # {domain_id: [orbit_indices, 1-based]}
    inter_domain_links: dict  # {(domain_a, domain_b): [(sat_in_a, sat_in_b), ...]}
    orbits_per_domain: int
    sats_per_orbit: int
    is_ring: bool = True  # True for Walker-Delta (ring), False for Walker-Star (chain)


def by_orbit_group(shell, n_domains):
    """Partition shell into n_domains by grouping consecutive orbital planes.

    Parameters
    ----------
    shell : shell object from StarPerf
        Must have: number_of_orbits, number_of_satellite_per_orbit, inclination
    n_domains : int
        Number of domains. Must evenly divide number_of_orbits.

    Returns
    -------
    DomainInfo
        Complete domain partition information.

    Raises
    ------
    NotImplementedError
        If shell inclination is in the polar range (80° < inc < 100°).
    """
    n_orbits = shell.number_of_orbits
    sats_per_orbit = shell.number_of_satellite_per_orbit
    is_polar = 80 < shell.inclination < 100

    if is_polar:
        raise NotImplementedError(
            f"by_orbit_group currently only supports Walker-Delta (inclination <= 80°), "
            f"but shell inclination = {shell.inclination}°. "
            f"For Walker-Star polar constellations, use by_orbit_group_polar plugin (TODO)."
        )

    assert n_orbits % n_domains == 0, (
        f"number_of_orbits ({n_orbits}) must be divisible by n_domains ({n_domains})"
    )
    orbits_per_domain = n_orbits // n_domains

    domain_of_sat = {}
    domain_sats = {d: [] for d in range(n_domains)}
    domain_orbits = {d: [] for d in range(n_domains)}

    for oi in range(1, n_orbits + 1):
        domain_id = (oi - 1) // orbits_per_domain
        domain_orbits[domain_id].append(oi)
        for si in range(1, sats_per_orbit + 1):
            sat_id = (oi - 1) * sats_per_orbit + si
            domain_of_sat[sat_id] = domain_id
            domain_sats[domain_id].append(sat_id)

    # Identify inter-domain links: inter-orbit ISLs at domain boundaries
    # Walker-Delta: ring topology — all adjacent domain pairs connected (including last↔first)
    inter_domain_links = {}
    for d in range(n_domains):
        d_next = (d + 1) % n_domains
        right_orbit = domain_orbits[d][-1]
        left_orbit = domain_orbits[d_next][0]
        links = []
        for si in range(1, sats_per_orbit + 1):
            sat_a = (right_orbit - 1) * sats_per_orbit + si
            sat_b = (left_orbit - 1) * sats_per_orbit + si
            links.append((sat_a, sat_b))
        inter_domain_links[(d, d_next)] = links

    return DomainInfo(
        n_domains=n_domains,
        domain_of_sat=domain_of_sat,
        domain_sats=domain_sats,
        domain_orbits=domain_orbits,
        inter_domain_links=inter_domain_links,
        orbits_per_domain=orbits_per_domain,
        sats_per_orbit=sats_per_orbit,
        is_ring=True,
    )
