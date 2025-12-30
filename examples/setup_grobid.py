#!/usr/bin/env python3
"""
Grobid Server Setup Helper

This script helps users set up and manage the Grobid server for enhanced PDF processing.
"""

import subprocess
import time
import requests
import sys
from pathlib import Path

def check_docker_available():
    """Check if Docker is available on the system"""
    try:
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False

def check_grobid_server(url="http://localhost:8070"):
    """Check if Grobid server is running"""
    try:
        response = requests.get(f"{url}/api/isalive", timeout=5)
        return response.status_code == 200 and response.text.strip() == 'true'
    except:
        return False

def start_grobid_docker():
    """Start Grobid server using Docker"""
    print("🐳 Starting Grobid server with Docker...")
    print("This may take a few minutes for first-time setup...")
    
    try:
        # Run Grobid in Docker
        cmd = [
            'docker', 'run', '-t', '--rm', '-p', '8070:8070',
            'lfoppiano/grobid:0.8.0'
        ]
        
        print(f"Running: {' '.join(cmd)}")
        print("Press Ctrl+C to stop the server")
        
        # Start the process
        process = subprocess.Popen(cmd)
        
        # Wait a bit for server to start
        print("\n⏳ Waiting for server to start...")
        for i in range(30):  # Wait up to 30 seconds
            if check_grobid_server():
                print("✅ Grobid server is running!")
                print("You can now process academic papers with structure extraction.")
                break
            time.sleep(1)
            if i % 5 == 0:
                print(f"   Still waiting... ({i+1}/30 seconds)")
        else:
            print("⚠️  Server might still be starting. Check http://localhost:8070/api/isalive")
        
        # Keep the process running
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Stopping Grobid server...")
            process.terminate()
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting Docker container: {e}")
        return False
    except KeyboardInterrupt:
        print("\n🛑 Cancelled by user")
        return False
    
    return True

def show_manual_instructions():
    """Show manual installation instructions"""
    print("\n📖 Manual Grobid Setup Instructions:")
    print("="*50)
    print("1. Prerequisites:")
    print("   - Java 8 or higher")
    print("   - Git")
    print()
    print("2. Clone and build Grobid:")
    print("   git clone https://github.com/kermitt2/grobid.git")
    print("   cd grobid")
    print("   ./gradlew run")
    print()
    print("3. Alternative with Gradle wrapper (Windows):")
    print("   gradlew.bat run")
    print()
    print("4. The server will start on http://localhost:8070")
    print()
    print("5. Verify with:")
    print("   curl http://localhost:8070/api/isalive")
    print("   (should return 'true')")

def main():
    print("🔬 Grobid Server Setup Helper")
    print("="*50)
    
    # Check if server is already running
    if check_grobid_server():
        print("✅ Grobid server is already running!")
        print("You can proceed with document processing.")
        return
    
    print("Grobid server is not running.")
    print("\nSetup options:")
    print("1. 🐳 Start with Docker (recommended)")
    print("2. 📖 Show manual installation instructions")
    print("3. 🔍 Check server status only")
    print("4. ❌ Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == '1':
                if not check_docker_available():
                    print("❌ Docker is not available on this system.")
                    print("Please install Docker or use manual installation (option 2).")
                    continue
                
                print("\n🚀 Starting Grobid with Docker...")
                success = start_grobid_docker()
                if success:
                    print("✅ Grobid server setup completed!")
                break
                
            elif choice == '2':
                show_manual_instructions()
                break
                
            elif choice == '3':
                print("\n🔍 Checking server status...")
                if check_grobid_server():
                    print("✅ Grobid server is running!")
                else:
                    print("❌ Grobid server is not running.")
                break
                
            elif choice == '4':
                print("👋 Exiting...")
                break
                
            else:
                print("❌ Invalid choice. Please enter 1-4.")
                
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            break

def quick_test():
    """Quick test of the enhanced document loader"""
    print("\n🧪 Quick Test of Enhanced Document Loader")
    print("-"*40)
    
    try:
        sys.path.append(str(Path(__file__).parent / 'src'))
        from document_loader import DocumentLoader
        
        # Test with Grobid enabled
        loader = DocumentLoader(use_grobid=True)
        status = loader.get_grobid_status()
        
        print(f"Grobid Support: {status['grobid_support']}")
        print(f"Server Available: {status['server_available']}")
        
        if status['server_available']:
            print("✅ Ready to process academic papers with Grobid!")
        else:
            print("⚠️  Grobid server not available - will use PyPDF2 fallback")
            
    except ImportError:
        print("❌ Cannot import document_loader module")
    except Exception as e:
        print(f"❌ Error during test: {e}")

if __name__ == "__main__":
    try:
        main()
        quick_test()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)