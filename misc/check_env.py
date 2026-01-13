import api


def main():
    print("AUTH set:", bool(api.GIGACHAT_BASIC_AUTH))
    print("AUTH len:", len(api.GIGACHAT_BASIC_AUTH or ""))
    print("MODEL:", api.GIGACHAT_MODEL)
    print("SCOPE:", api.GIGACHAT_SCOPE)
    print("API BASE:", api.GIGACHAT_API_BASE)
    print("AUTH value:", api.GIGACHAT_BASIC_AUTH)


if __name__ == "__main__":
    main()
