"""
CU-2: Sector & Country Mapping

Provides look-up tables for HS → sector, country code → name,
and segment tags (soybeans, autos, semiconductors).
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import logging
from .config import DATA_EXTERNAL

logger = logging.getLogger(__name__)


# === HS Code Mapping Constants ===

# Soybeans HS codes
SOYBEANS_HS = [
    '1201',  # Soybeans, whether or not broken
    '120100',
    '12019000',
    '12011000',
]

# Automobiles HS codes (Chapter 87)
AUTOS_HS = [
    '8703',  # Motor cars and other motor vehicles for transport of persons
    '870310',
    '870321',
    '870322',
    '870323',
    '870324',
    '870331',
    '870332',
    '870333',
]

# Semiconductors HS codes (Chapter 85.41, 85.42)
SEMICONDUCTORS_HS = {
    'high': [
        '854140',  # Photosensitive semiconductor devices, incl. photovoltaic cells
        '854141',
        '854142',
        '854143',
    ],
    'mid': [
        '854231',  # Processors and controllers
        '854232',
        '854233',
    ],
    'low': [
        '854290',  # Other electronic integrated circuits
        '854110',
        '854121',
    ],
}

# Country name standardization
COUNTRY_NAME_MAP = {
    'China': 'China',
    'Brazil': 'Brazil',
    'Argentina': 'Argentina',
    'Japan': 'Japan',
    'Mexico': 'Mexico',
    'Canada': 'Canada',
    'Germany': 'Germany',
    'South Korea': 'South Korea',
    'Taiwan': 'Taiwan',
    'United States': 'United States',
    'European Union': 'European Union',
}


class HSMapper:
    """Mapper for HS codes to sectors and segments."""
    
    def __init__(self):
        """Initialize the mapper with predefined mappings."""
        self.soybeans_hs = SOYBEANS_HS
        self.autos_hs = AUTOS_HS
        self.semiconductors_hs = SEMICONDUCTORS_HS
        
    def is_soybean(self, hs_code: str) -> bool:
        """Check if HS code is for soybeans.
        
        Args:
            hs_code: HS code (2, 4, 6, 8, or 10 digits)
            
        Returns:
            True if hs_code matches soybeans
        """
        hs_str = str(hs_code).strip()
        for soy_code in self.soybeans_hs:
            if hs_str.startswith(soy_code):
                return True
        return False
    
    def is_auto(self, hs_code: str) -> bool:
        """Check if HS code is for automobiles.
        
        Args:
            hs_code: HS code
            
        Returns:
            True if hs_code matches automobiles
        """
        hs_str = str(hs_code).strip()
        for auto_code in self.autos_hs:
            if hs_str.startswith(auto_code):
                return True
        return False
    
    def get_semiconductor_segment(self, hs_code: str) -> Optional[str]:
        """Get semiconductor segment for HS code.
        
        Args:
            hs_code: HS code
            
        Returns:
            Segment name ('high', 'mid', 'low') or None
        """
        hs_str = str(hs_code).strip()
        
        for segment, codes in self.semiconductors_hs.items():
            for code in codes:
                if hs_str.startswith(code):
                    return segment
        
        return None
    
    def is_semiconductor(self, hs_code: str) -> bool:
        """Check if HS code is for semiconductors.
        
        Args:
            hs_code: HS code
            
        Returns:
            True if hs_code matches semiconductors
        """
        return self.get_semiconductor_segment(hs_code) is not None
    
    def tag_dataframe(self, df: pd.DataFrame, hs_column: str = 'hs_code') -> pd.DataFrame:
        """Add product category tags to a DataFrame.
        
        Args:
            df: DataFrame with HS codes
            hs_column: Name of the HS code column
            
        Returns:
            DataFrame with added columns: is_soybean, is_auto, 
            is_semiconductor, semiconductor_segment
        """
        df = df.copy()
        
        df['is_soybean'] = df[hs_column].apply(self.is_soybean)
        df['is_auto'] = df[hs_column].apply(self.is_auto)
        df['is_semiconductor'] = df[hs_column].apply(self.is_semiconductor)
        df['semiconductor_segment'] = df[hs_column].apply(self.get_semiconductor_segment)
        
        return df


class CountryMapper:
    """Mapper for country codes and names."""
    
    def __init__(self):
        """Initialize with standard country name map."""
        self.name_map = COUNTRY_NAME_MAP
    
    def standardize_name(self, country: str) -> str:
        """Standardize country name.
        
        Args:
            country: Raw country name
            
        Returns:
            Standardized country name
        """
        country_clean = str(country).strip()
        return self.name_map.get(country_clean, country_clean)
    
    def tag_dataframe(
        self,
        df: pd.DataFrame,
        country_column: str = 'partner_country'
    ) -> pd.DataFrame:
        """Standardize country names in a DataFrame.
        
        Args:
            df: DataFrame with country column
            country_column: Name of the country column
            
        Returns:
            DataFrame with standardized country names
        """
        df = df.copy()
        df[f'{country_column}_standardized'] = df[country_column].apply(
            self.standardize_name
        )
        return df


def create_hs_sector_mapping() -> pd.DataFrame:
    """Create a comprehensive HS to sector mapping table.
    
    Returns:
        DataFrame with columns: hs_code, sector, subsector, segment
    """
    mapper = HSMapper()
    
    # Build mapping table
    mappings = []
    
    # Soybeans
    for hs in SOYBEANS_HS:
        mappings.append({
            'hs_code': hs,
            'sector': 'Agriculture',
            'subsector': 'Soybeans',
            'segment': None
        })
    
    # Automobiles
    for hs in AUTOS_HS:
        mappings.append({
            'hs_code': hs,
            'sector': 'Manufacturing',
            'subsector': 'Automobiles',
            'segment': None
        })
    
    # Semiconductors
    for segment, codes in SEMICONDUCTORS_HS.items():
        for hs in codes:
            mappings.append({
                'hs_code': hs,
                'sector': 'Technology',
                'subsector': 'Semiconductors',
                'segment': segment
            })
    
    df = pd.DataFrame(mappings)
    
    return df


def save_mapping_tables() -> None:
    """Create and save mapping tables to DATA_EXTERNAL."""
    logger.info("Creating and saving mapping tables")
    
    # HS to sector mapping
    hs_mapping = create_hs_sector_mapping()
    
    # Soybeans
    soybeans_df = pd.DataFrame({
        'hs_code': SOYBEANS_HS,
        'product': 'Soybeans'
    })
    soybeans_df.to_csv(DATA_EXTERNAL / 'hs_soybeans.csv', index=False)
    logger.info(f"Saved soybeans mapping to {DATA_EXTERNAL / 'hs_soybeans.csv'}")
    
    # Autos
    autos_df = pd.DataFrame({
        'hs_code': AUTOS_HS,
        'product': 'Automobiles'
    })
    autos_df.to_csv(DATA_EXTERNAL / 'hs_autos.csv', index=False)
    logger.info(f"Saved autos mapping to {DATA_EXTERNAL / 'hs_autos.csv'}")
    
    # Semiconductors with segments
    semi_records = []
    for segment, codes in SEMICONDUCTORS_HS.items():
        for code in codes:
            semi_records.append({
                'hs_code': code,
                'segment': segment,
                'product': 'Semiconductors'
            })
    semi_df = pd.DataFrame(semi_records)
    semi_df.to_csv(DATA_EXTERNAL / 'hs_semiconductors_segmented.csv', index=False)
    logger.info(f"Saved semiconductors mapping to {DATA_EXTERNAL / 'hs_semiconductors_segmented.csv'}")
    
    # Full sector mapping
    hs_mapping.to_csv(DATA_EXTERNAL / 'hs_to_sector.csv', index=False)
    logger.info(f"Saved full sector mapping to {DATA_EXTERNAL / 'hs_to_sector.csv'}")
    
    logger.info("All mapping tables saved successfully")


if __name__ == '__main__':
    # Create and save mapping tables when run as script
    save_mapping_tables()
