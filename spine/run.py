from spine.dataloader.dataloader import DataLoader


def main():
    loader = DataLoader().load_default()
    spines = loader.spines
    for spine in spines:
        print(spine.data.shape)


if __name__ == "__main__":
    main()
