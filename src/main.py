import sys

def main():
    print("=" * 64)
    print("CONTENT-BASED IMAGE RETRIEVAL SYSTEM")
    print("=" * 64)
    
    while True:
        print("\n1. Extract shape features")
        print("2. Extract texture features")
        print("3. Search by shape")
        print("4. Search by texture")
        print("0. Exit")
        
        choice = input("\nChoice: ").strip()
        
        if choice == '1':
            process_all_shape_images("data/Formes", "features/Formes")
            
        elif choice == '2':
            process_all_texture_images("data/Textures", "features/Textures")
            
        elif choice == '3':
            query = input("Query image name: ").strip()
            try:
                results = retrieve_similar_shapes(query, "features/Formes", "data/Formes", 6)
                print("\nResults:")
                for i, (name, dist, path) in enumerate(results, 1):
                    print(f"{i}. {name} - Distance: {dist:.6f}")
                
                viz = input("\nVisualize? (y/n): ").strip().lower()
                if viz == 'y':
                    query_path = os.path.join("data/Formes", query)
                    output_path = f"results/shape_results/result_{Path(query).stem}.png"
                    visualize_shape_results(query_path, results, output_path)
            except Exception as e:
                print(f"Error: {e}")
                
        elif choice == '4':
            query = input("Query image name: ").strip()
            try:
                results = retrieve_similar_textures(query, "features/Textures", "data/Textures", 6)
                print("\nResults:")
                for i, (name, dist, path) in enumerate(results, 1):
                    print(f"{i}. {name} - Distance: {dist:.6f}")
                
                viz = input("\nVisualize? (y/n): ").strip().lower()
                if viz == 'y':
                    query_path = os.path.join("data/Textures", query)
                    output_path = f"results/texture_results/result_{Path(query).stem}.png"
                    visualize_texture_results(query_path, results, output_path)
            except Exception as e:
                print(f"Error: {e}")
                
        elif choice == '0':
            print("Exiting...")
            break
            
        else:
            print("Invalid choice")

if __name__ == "__main__":
        main()