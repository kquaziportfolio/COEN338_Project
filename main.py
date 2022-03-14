import click
import lib_kcompress


@click.command()
@click.argument("infp", type=click.Path(exists=True))
@click.argument("outfp")
@click.argument("ccount", type=int, required=False)
def main(infp, outfp, ccount):
    if ccount is None:
        ccount = 160
    lib_kcompress.compress(infp, outfp, ccount)


main()
