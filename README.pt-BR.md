# foresight (português)

> Versão resumida — a documentação completa e sempre atual está no [README em inglês](README.md).

Previsão de séries temporais em Rust que escolhe o modelo pelo que teria
funcionado. O `foresight` ajusta vários modelos à série, refaz o passado para
ver como cada um teria se saído (backtest por origem móvel), escolhe pelo erro
fora da amostra e informa intervalos tirados dos erros de fato observados. Não
tem dependências e todo resultado é determinístico.

**Site:** <https://milkway.github.io/foresight/>

## Instalação

```bash
cargo add --git https://github.com/milkway/foresight foresight
```

## Uso básico

```rust
use foresight::{models, Backtest, Series};

// dados mensais cuja primeira observação é de março
let y = Series::monthly(&valores, 2);
let relatorio = Backtest::default().run(y, &models::defaults()).unwrap();

let melhor = relatorio.chosen();
for p in &melhor.forecast {
    let faixa = p.interval(0.80).unwrap();
    println!("{} {:.0} [{:.0}, {:.0}]", p.horizon, p.mean, faixa.lower, faixa.upper);
}

// total dos próximos seis meses, com faixa própria
let semestre = melhor.cumulative(6).unwrap();
```

## O que tem

- Modelos: média, ingênuo, tendência, sazonal ingênuo, Theta, Holt-Winters,
  regressão log-linear (com deflator opcional), ARIMA sazonal por máxima
  verossimilhança exata, ARIMA com ordens automáticas, a família ETS de
  suavização exponencial com escolha automática e Prophet (tendência com
  pontos de quebra, sazonalidade de Fourier, eventos datados e degraus), sem Stan.
- Qualquer modelo na escala log ou Box-Cox (`Transformed`).
- Backtest por origem móvel em todos os núcleos, com MAPE, MAE, RMSE, MASE e
  viés por horizonte, média dos melhores modelos e escolha pelo erro fora da
  amostra.
- Intervalos empíricos por horizonte e para totais acumulados.
- Diagnósticos: teste KPSS, número de diferenças, força da sazonalidade.

Os métodos foram implementados a partir dos artigos e conferidos com os pacotes
`forecast` e `prophet` do R em dados públicos (`tests/against_r.rs`).

## Licença

MIT. Veja [LICENSE-MIT](LICENSE-MIT).
