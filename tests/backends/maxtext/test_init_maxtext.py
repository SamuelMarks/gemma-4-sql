import gemma_4_sql.backends.maxtext as m_init


def test_init_get_trainer():
    assert m_init.get_trainer() == "maxtext_trainer"
