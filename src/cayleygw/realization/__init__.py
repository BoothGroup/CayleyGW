"""Realization of matrix Cayley moments as one sector's poles and couplings.

:mod:`.block_cmv` and :mod:`.toeplitz` build a finite unitary that reproduces
the moments, and :mod:`.sector` picks its terminal closure and maps its
eigenvalues to real poles. The subpackage knows only a moment sequence, a
sector and a Cayley map, and imports nothing from PySCF.
"""
