"""
Fizikalne konstante i simulacijski parametri za simulaciju ulaska u spiralu
binarnog sustava crnih rupa GW150914. Sve veličine u SI jedinicama, osim
ako nije drugačije naznačeno.
"""

G      = 6.674e-11    # gravitacijska konstanta [m³ kg⁻¹ s⁻²]
C      = 2.998e8      # brzina svjetlosti [m/s]
M_SUN  = 1.989e30     # sunčeva masa [kg]
MPC    = 3.086e22     # megaparsek [m]

# Parametri za simulaciju GW150914. Mase i udaljenost prema Abbott et al. (2016).
# Frekvencija uzorkovanja 4096 Hz pokriva pojas 20–2048 Hz.
GW150914 = dict(
    name        = "GW150914",
    m1          = 35.6,
    m2          = 30.6,
    d_mpc       = 440.0,
    iota_deg    = 153.0,
    f0_hz       = 20.0,
    sample_rate = 4096,
    label       = r"GW150914 ($m_1=35.6\,M_\odot,\;m_2=30.6\,M_\odot,\;d=440\,$Mpc)",
)

PARAMS = GW150914.copy()

# Varijante masa za studiju osjetljivosti. Sve s istom udaljenošću (440 Mpc)
# i inklinacijom 0° (lice-prema-promatraču) kako bi se izolirao učinak mase.
MASS_VARIANTS = [
    dict(m1=10.0,  m2=10.0,  d_mpc=440, iota_deg=0, f0_hz=20, label=r"$10+10\,M_\odot$"),
    dict(m1=20.0,  m2=20.0,  d_mpc=440, iota_deg=0, f0_hz=20, label=r"$20+20\,M_\odot$"),
    dict(m1=35.6,  m2=30.6,  d_mpc=440, iota_deg=0, f0_hz=20, label=r"$35.6+30.6\,M_\odot$ (GW150914)"),
    dict(m1=60.0,  m2=50.0,  d_mpc=440, iota_deg=0, f0_hz=10, label=r"$60+50\,M_\odot$"),
]

# Varijante udaljenosti za studiju osjetljivosti. Iste mase kao GW150914, inklinacija 0°.
DISTANCE_VARIANTS = [
    dict(m1=35.6, m2=30.6, d_mpc=100,  iota_deg=0, f0_hz=20, label=r"$d=100\,$Mpc"),
    dict(m1=35.6, m2=30.6, d_mpc=440,  iota_deg=0, f0_hz=20, label=r"$d=440\,$Mpc (GW150914)"),
    dict(m1=35.6, m2=30.6, d_mpc=1000, iota_deg=0, f0_hz=20, label=r"$d=1000\,$Mpc"),
    dict(m1=35.6, m2=30.6, d_mpc=2000, iota_deg=0, f0_hz=20, label=r"$d=2000\,$Mpc"),
]

# Varijante kuta inklinacije za studiju polarizacijske ovisnosti.
# Iste mase i udaljenost kao GW150914.
INCLINATION_VARIANTS = [
    dict(m1=35.6, m2=30.6, d_mpc=440, iota_deg=0,   f0_hz=20, label=r"$\iota=0°$ (face-on)"),
    dict(m1=35.6, m2=30.6, d_mpc=440, iota_deg=45,  f0_hz=20, label=r"$\iota=45°$"),
    dict(m1=35.6, m2=30.6, d_mpc=440, iota_deg=90,  f0_hz=20, label=r"$\iota=90°$ (edge-on)"),
    dict(m1=35.6, m2=30.6, d_mpc=440, iota_deg=153, f0_hz=20, label=r"$\iota=153°$ (GW150914)"),
]
