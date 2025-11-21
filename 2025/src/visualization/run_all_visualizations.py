import sys
from pathlib import Path
import logging

# Add src to path
src_dir = Path(__file__).parents[1]
sys.path.append(str(src_dir))

from models.q1_soybeans import SoybeanTradeModel
from models.q2_autos import AutoTradeModel
from models.q3_semiconductors import SemiconductorModel
from models.q4_tariff_revenue import TariffRevenueModel
from models.q5_macro_finance import MacroFinanceModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_all():
    logger.info("Starting full visualization generation...")
    
    # Q1
    try:
        logger.info("Generating Q1 visualizations...")
        q1 = SoybeanTradeModel()
        # Load data and run simulation to get results for plotting
        q1.load_q1_data()
        q1.prepare_panel_for_estimation()
        q1.estimate_trade_elasticities()
        results = q1.simulate_tariff_scenarios()
        q1.plot_q1_results(results)
    except Exception as e:
        logger.error(f"Q1 Visualization failed: {e}")

    # Q2
    try:
        logger.info("Generating Q2 visualizations...")
        q2 = AutoTradeModel()
        q2.load_q2_data()
        q2.load_external_auto_data()
        q2.estimate_import_structure_model()
        q2.estimate_industry_transmission_model()
        # We need simulation results to plot/analyze
        q2.simulate_japan_response_scenarios()
        # Note: Q2 plotting is embedded in the analysis report generation or separate visualizer
        # The current code generates CSVs and JSONs. 
        # We might need to add specific plotting calls if they aren't auto-generated.
        # Checking q2_autos.py, it generates 'analysis_report.md' but no direct matplotlib plots in the main flow 
        # other than what might be in a separate visualizer. 
        # Let's check if we can trigger any plots. 
        # It seems q2_autos.py focuses on data export. 
        pass 
    except Exception as e:
        logger.error(f"Q2 Visualization failed: {e}")

    # Q3
    try:
        logger.info("Generating Q3 visualizations...")
        q3 = SemiconductorModel()
        q3.load_q3_data()
        q3.load_external_chip_data()
        q3.estimate_output_response()
        q3.simulate_policy_combinations()
        q3.plot_q3_results()
    except Exception as e:
        logger.error(f"Q3 Visualization failed: {e}")

    # Q4
    try:
        logger.info("Generating Q4 visualizations...")
        q4 = TariffRevenueModel()
        q4.load_q4_data()
        q4.estimate_static_revenue_model()
        q4.estimate_dynamic_import_response()
        q4.simulate_second_term_revenue()
        q4.plot_q4_results()
    except Exception as e:
        logger.error(f"Q4 Visualization failed: {e}")

    # Q5
    try:
        logger.info("Generating Q5 visualizations...")
        q5 = MacroFinanceModel()
        q5.load_q5_data()
        q5.estimate_var_model() # This generates IRF data
        q5.plot_q5_results()
    except Exception as e:
        logger.error(f"Q5 Visualization failed: {e}")

    logger.info("Visualization generation complete.")

if __name__ == "__main__":
    run_all()
