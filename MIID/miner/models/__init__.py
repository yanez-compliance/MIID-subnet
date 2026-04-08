# MIID/miner/models/__init__.py
#
# Model modules for image variation generation.
# Each module exposes:
#   load_pipeline(device, token)  -> pipeline object
#   generate(pipe, image, prompt) -> PIL Image
#   main()                        -> standalone test entrypoint
