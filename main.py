import os
import sys
from pathlib import Path

# Import all required functions
from src.utils import load_features_from_json, save_features_to_json, load_image
from src.shape_features import extract_shape_features, process_all_shape_images
from src.texture_features import extract_texture_features, process_all_texture_images
from src.shape_retrieval import retrieve_similar_shapes, visualize_shape_results
from src.texture_retrieval import retrieve_similar_textures, visualize_texture_results


def main():
    print("=" * 60)
    print("CONTENT-BASED IMAGE RETRIEVAL SYSTEM")
    print("=" * 60)
    
    while True:
        print("\n1. Extract shape features")
        print("2. Extract texture features")
        print("3. Search by shape")
        print("4. Search by texture")
        print("0. Exit")
        
        choice = input("\nChoice: ").strip()
        
        if choice == '1':
            print("\nExtracting shape features...")
            try:
                process_all_shape_images("data/Formes", "features/Formes")
                print("Shape features extracted successfully.")
            except Exception as e:
                print(f"Error: {e}")
            
        elif choice == '2':
            print("\nExtracting texture features...")
            try:
                process_all_texture_images("data/Textures", "features/Textures")
                print("Texture features extracted successfully.")
            except Exception as e:
                print(f"Error: {e}")
            
        elif choice == '3':
            query = input("\nQuery image name (e.g., apple-1.gif): ").strip()
            if not query:
                print("No image specified.")
                continue
            
            # Remove any path prefix if user provided full path
            query = os.path.basename(query)
                
            try:
                print(f"Searching for images similar to: {query}")
                results = retrieve_similar_shapes(
                    query, 
                    "features/Formes", 
                    "data/Formes", 
                    6
                )
                
                print("\nResults:")
                print("-" * 60)
                for i, (name, dist, path) in enumerate(results, 1):
                    similarity = max(0, 100 - dist * 10)
                    print(f"{i}. {name:20s} Distance: {dist:.6f}  Similarity: {similarity:.1f}%")
                
                viz = input("\nVisualize results? (y/n): ").strip().lower()
                if viz == 'y':
                    query_path = os.path.join("data/Formes", query)
                    output_path = os.path.join(
                        "results/shape_results", 
                        f"result_{Path(query).stem}.png"
                    )
                    visualize_shape_results(query_path, results, output_path)
                    
            except FileNotFoundError as e:
                print(f"Error: Image or features not found - {e}")
                print("Make sure to extract features first (option 1).")
            except Exception as e:
                print(f"Error: {e}")
                
        elif choice == '4':
            query = input("\nQuery image name (e.g., Im01.jpg): ").strip()
            if not query:
                print("No image specified.")
                continue
            
            # Remove any path prefix if user provided full path
            query = os.path.basename(query)
                
            try:
                print(f"Searching for images similar to: {query}")
                results = retrieve_similar_textures(
                    query, 
                    "features/Textures", 
                    "data/Textures", 
                    6
                )
                
                print("\nResults:")
                print("-" * 60)
                for i, (name, dist, path) in enumerate(results, 1):
                    similarity = max(0, 100 - dist * 20)
                    print(f"{i}. {name:20s} Distance: {dist:.6f}  Similarity: {similarity:.1f}%")
                
                viz = input("\nVisualize results? (y/n): ").strip().lower()
                if viz == 'y':
                    query_path = os.path.join("data/Textures", query)
                    output_path = os.path.join(
                        "results/texture_results", 
                        f"result_{Path(query).stem}.png"
                    )
                    visualize_texture_results(query_path, results, output_path)
                    
            except FileNotFoundError as e:
                print(f"Error: Image or features not found - {e}")
                print("Make sure to extract features first (option 2).")
            except Exception as e:
                print(f"Error: {e}")
                
        elif choice == '0':
            print("\nExiting...")
            break
            
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user.")
    except Exception as e:
        print(f"\nCritical error: {e}")
        sys.exit(1)