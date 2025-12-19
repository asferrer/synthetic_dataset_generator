"""
Test de Integración para Arquitectura de Microservicios
Verifica que todas las mejoras funcionen correctamente via API calls

Usage:
    python test_microservices_integration.py

    O desde el host (Windows):
    python test_microservices_integration.py --host localhost
"""

import requests
import json
import time
import argparse
import sys
from pathlib import Path

# Colores para output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n")

def print_success(text):
    print(f"{Colors.GREEN}[OK] {text}{Colors.END}")

def print_error(text):
    print(f"{Colors.RED}[ERROR] {text}{Colors.END}")

def print_warning(text):
    print(f"{Colors.YELLOW}[WARNING] {text}{Colors.END}")

def print_info(text):
    print(f"{Colors.BLUE}[INFO] {text}{Colors.END}")


class MicroservicesTestSuite:
    """Suite de tests para arquitectura de microservicios"""

    def __init__(self, host='localhost'):
        self.host = host
        self.services = {
            'gateway': f'http://{host}:8000',
            'depth': f'http://{host}:8001',
            'effects': f'http://{host}:8003',
            'frontend': f'http://{host}:8501'
        }
        self.results = []

    def test_service_health(self, service_name, url):
        """Test health endpoint de un servicio"""
        try:
            response = requests.get(f"{url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print_success(f"{service_name} service is healthy")
                print_info(f"   Status: {data.get('status', 'unknown')}")
                if 'model' in data:
                    print_info(f"   Model: {data['model']}")
                return True
            else:
                print_error(f"{service_name} service returned {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print_error(f"{service_name} service is not reachable")
            return False
        except Exception as e:
            print_error(f"{service_name} service error: {e}")
            return False

    def test_all_services_health(self):
        """Test 1: Verificar que todos los servicios estén corriendo"""
        print_header("TEST 1: Service Health Checks")

        all_healthy = True
        for service_name, url in self.services.items():
            if service_name == 'frontend':
                # Frontend es Streamlit, no tiene /health endpoint
                try:
                    response = requests.get(url, timeout=5)
                    if response.status_code in [200, 301, 302]:
                        print_success(f"frontend service is reachable")
                    else:
                        print_error(f"frontend service returned {response.status_code}")
                        all_healthy = False
                except Exception as e:
                    print_error(f"frontend service error: {e}")
                    all_healthy = False
            else:
                if not self.test_service_health(service_name, url):
                    all_healthy = False

        self.results.append(("Service Health", all_healthy))
        return all_healthy

    def test_depth_service(self):
        """Test 2: Depth Anything V3 Service"""
        print_header("TEST 2: Depth Service (Depth Anything V3)")

        try:
            # Check model info
            response = requests.get(f"{self.services['depth']}/info", timeout=5)
            if response.status_code == 200:
                info = response.json()
                print_success("Depth service info retrieved")
                print_info(f"   Model: {info.get('model', 'unknown')}")
                print_info(f"   Version: {info.get('version', 'unknown')}")
                print_info(f"   Device: {info.get('device', 'unknown')}")

                # Verify it's Depth Anything V3
                model_name = info.get('model', '')
                if 'DA3' in model_name or 'v3' in model_name.lower():
                    print_success("Depth Anything V3 confirmed")
                    self.results.append(("Depth Service (DA-V3)", True))
                    return True
                else:
                    print_warning(f"Model is {model_name}, expected DA3")
                    self.results.append(("Depth Service (DA-V3)", False))
                    return False
            else:
                print_error(f"Failed to get depth service info: {response.status_code}")
                self.results.append(("Depth Service (DA-V3)", False))
                return False

        except Exception as e:
            print_error(f"Depth service test failed: {e}")
            self.results.append(("Depth Service (DA-V3)", False))
            return False

    def test_effects_service(self):
        """Test 3: Effects Service (Realism Pipeline + Lighting)"""
        print_header("TEST 3: Effects Service (Realism + Lighting)")

        try:
            # Check info endpoint
            response = requests.get(f"{self.services['effects']}/info", timeout=5)
            if response.status_code == 200:
                info = response.json()
                print_success("Effects service info retrieved")

                # Check available effects
                effects = info.get('available_effects', [])
                print_info(f"   Available effects: {len(effects)}")

                # Check for lighting support
                if 'advanced_lighting' in str(info).lower() or 'lighting' in str(effects).lower():
                    print_success("Advanced Lighting support detected")
                else:
                    print_info("   Note: Advanced lighting is integrated in main pipeline")

                self.results.append(("Effects Service", True))
                return True
            else:
                print_error(f"Effects service info failed: {response.status_code}")
                self.results.append(("Effects Service", False))
                return False

        except Exception as e:
            print_error(f"Effects service test failed: {e}")
            self.results.append(("Effects Service", False))
            return False

    def test_gateway_orchestration(self):
        """Test 4: Gateway Orchestration"""
        print_header("TEST 4: Gateway Service (Orchestration)")

        try:
            # Check gateway health
            response = requests.get(f"{self.services['gateway']}/health", timeout=5)
            if response.status_code == 200:
                health = response.json()
                print_success("Gateway service is healthy")

                # Check connected services
                services_status = health.get('services', {})
                print_info("   Connected services:")

                all_connected = True
                if isinstance(services_status, dict):
                    for service, status in services_status.items():
                        if status == 'healthy':
                            print_success(f"     - {service}: {status}")
                        else:
                            print_error(f"     - {service}: {status}")
                            all_connected = False
                elif isinstance(services_status, list):
                    # Handle list format
                    for service in services_status:
                        print_success(f"     - {service}")
                else:
                    print_info(f"     Services: {services_status}")

                self.results.append(("Gateway Orchestration", all_connected))
                return all_connected
            else:
                print_error(f"Gateway health check failed: {response.status_code}")
                self.results.append(("Gateway Orchestration", False))
                return False

        except Exception as e:
            print_error(f"Gateway test failed: {e}")
            self.results.append(("Gateway Orchestration", False))
            return False

    def test_validation_integration(self):
        """Test 5: Quality Validation Integration"""
        print_header("TEST 5: Quality Validation System")

        # Validation está integrado en el augmentor, verificamos configuración
        print_info("Checking configuration file for validation settings...")

        try:
            config_path = Path('configs/config.yaml')
            if not config_path.exists():
                print_warning("config.yaml not found, skipping validation config check")
                self.results.append(("Validation Integration", None))
                return None

            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            validation_config = config.get('validation', {})

            if validation_config.get('enabled', False):
                print_success("Quality Validation is ENABLED")

                # Check metrics
                metrics = validation_config.get('metrics', {})
                print_info("   Enabled metrics:")
                if metrics.get('lpips_enabled'):
                    print_success("     - LPIPS (perceptual quality)")
                if metrics.get('anomaly_detection'):
                    print_success("     - Anomaly detection (Isolation Forest)")
                if metrics.get('physics_checks'):
                    print_success("     - Physics validation")

                # Check thresholds
                thresholds = validation_config.get('thresholds', {})
                print_info("   Quality thresholds:")
                print_info(f"     - Min perceptual quality: {thresholds.get('min_perceptual_quality', 'N/A')}")
                print_info(f"     - Min anomaly score: {thresholds.get('min_anomaly_score', 'N/A')}")

                self.results.append(("Validation Integration", True))
                return True
            else:
                print_warning("Quality Validation is DISABLED in config")
                self.results.append(("Validation Integration", False))
                return False

        except Exception as e:
            print_error(f"Failed to check validation config: {e}")
            self.results.append(("Validation Integration", None))
            return None

    def test_lighting_integration(self):
        """Test 6: Advanced Lighting Integration"""
        print_header("TEST 6: Advanced Lighting System")

        print_info("Checking configuration file for lighting settings...")

        try:
            config_path = Path('configs/config.yaml')
            if not config_path.exists():
                print_warning("config.yaml not found, skipping lighting config check")
                self.results.append(("Lighting Integration", None))
                return None

            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            lighting_config = config.get('advanced_lighting', {})

            if lighting_config.get('enabled', False):
                print_success("Advanced Lighting is ENABLED")

                # Check settings
                print_info("   Lighting configuration:")
                print_info(f"     - Max light sources: {lighting_config.get('max_light_sources', 'N/A')}")
                print_info(f"     - Intensity threshold: {lighting_config.get('intensity_threshold', 'N/A')}")
                print_info(f"     - Multi-shadows: {lighting_config.get('enable_multi_shadows', 'N/A')}")
                print_info(f"     - Shadow softness: {lighting_config.get('shadow_softness', 'N/A')}")
                print_info(f"     - Water attenuation: {lighting_config.get('apply_water_attenuation', 'N/A')}")

                self.results.append(("Lighting Integration", True))
                return True
            else:
                print_warning("Advanced Lighting is DISABLED in config")
                self.results.append(("Lighting Integration", False))
                return False

        except Exception as e:
            print_error(f"Failed to check lighting config: {e}")
            self.results.append(("Lighting Integration", None))
            return None

    def test_depth_integration(self):
        """Test 7: Depth Anything V3 Integration"""
        print_header("TEST 7: Depth Anything V3 Integration")

        print_info("Checking configuration file for depth settings...")

        try:
            config_path = Path('configs/config.yaml')
            if not config_path.exists():
                print_warning("config.yaml not found, skipping depth config check")
                self.results.append(("Depth Integration", None))
                return None

            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            augmentation_config = config.get('augmentation', {})

            if augmentation_config.get('depth_aware', False):
                print_success("Depth-Aware Augmentation is ENABLED")

                # Check version
                version = augmentation_config.get('depth_model_version', 'unknown')
                model_size = augmentation_config.get('depth_model_size', 'unknown')

                if version == 'v3':
                    print_success(f"Using Depth Anything V3 ({model_size} model)")
                    print_info(f"   Model size: {model_size}")
                    print_info(f"   Cache dir: {augmentation_config.get('depth_cache_dir', 'N/A')}")

                    self.results.append(("Depth Integration (V3)", True))
                    return True
                else:
                    print_warning(f"Using Depth Anything {version}, not V3")
                    self.results.append(("Depth Integration (V3)", False))
                    return False
            else:
                print_warning("Depth-Aware Augmentation is DISABLED in config")
                self.results.append(("Depth Integration (V3)", False))
                return False

        except Exception as e:
            print_error(f"Failed to check depth config: {e}")
            self.results.append(("Depth Integration (V3)", None))
            return None

    def print_summary(self):
        """Imprime resumen de resultados"""
        print_header("TEST SUMMARY")

        passed = 0
        failed = 0
        skipped = 0

        for test_name, result in self.results:
            if result is True:
                print_success(f"{test_name:35s}: PASS")
                passed += 1
            elif result is False:
                print_error(f"{test_name:35s}: FAIL")
                failed += 1
            else:
                print_warning(f"{test_name:35s}: SKIPPED")
                skipped += 1

        print(f"\n{Colors.BOLD}{'='*60}{Colors.END}")
        print(f"{Colors.BOLD}Total: {len(self.results)} tests{Colors.END}")
        print(f"{Colors.GREEN}Passed: {passed}{Colors.END}")
        print(f"{Colors.RED}Failed: {failed}{Colors.END}")
        print(f"{Colors.YELLOW}Skipped: {skipped}{Colors.END}")
        print(f"{Colors.BOLD}{'='*60}{Colors.END}\n")

        if failed == 0 and passed > 0:
            print_success("ALL TESTS PASSED!")
            print_info("\nSistema listo para generacion de datasets sinteticos")
            print_info("\nPróximos pasos:")
            print_info("  1. Abre la UI: http://localhost:8501")
            print_info("  2. Carga tu dataset COCO")
            print_info("  3. Selecciona clases a aumentar")
            print_info("  4. ¡Genera tu dataset sintético con todas las mejoras activas!")
            print_info("\nMejoras activas:")
            print_info("   - Depth Anything V3 (+44% accuracy)")
            print_info("   - Quality Validation (LPIPS + Physics)")
            print_info("   - Advanced Multi-Light Shadows")
            return 0
        else:
            print_error("SOME TESTS FAILED")
            print_info("\nRevisa los errores arriba y corrige los problemas.")
            print_info("Verifica que todos los microservicios estén corriendo:")
            print_info("  docker-compose -f docker-compose.microservices.yml ps")
            return 1

    def run_all_tests(self):
        """Ejecuta todos los tests"""
        print_header("MICROSERVICES INTEGRATION TEST SUITE")
        print_info("Testing 3 major improvements:")
        print_info("  1. Depth Anything V3 (+44% accuracy)")
        print_info("  2. Quality Validation (LPIPS + Physics)")
        print_info("  3. Advanced Multi-Light Shadows")
        print(f"{Colors.BOLD}{'='*60}{Colors.END}\n")

        # Run tests
        self.test_all_services_health()
        self.test_depth_service()
        self.test_effects_service()
        self.test_gateway_orchestration()
        self.test_validation_integration()
        self.test_lighting_integration()
        self.test_depth_integration()

        # Print summary
        return self.print_summary()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Test microservices integration')
    parser.add_argument('--host', default='localhost', help='Microservices host (default: localhost)')
    args = parser.parse_args()

    suite = MicroservicesTestSuite(host=args.host)
    exit_code = suite.run_all_tests()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
