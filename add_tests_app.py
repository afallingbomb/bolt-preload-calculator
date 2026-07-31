with open('tests/test_app.py', 'a') as f:
    f.write('''

def test_imperial_unit_switch():
    """D3: Test Imperial unit switch and rendering."""
    at = AppTest.from_file("app.py")
    at.run(timeout=15)
    
    # Toggle to Imperial
    at.toggle(key="unit_toggle").set_value(True).run()
    
    assert not at.exception
    # Verify standard imperial threads exist
    assert at.selectbox(key="bolt_size").value == '1/4"'
    
    # Switch back to metric
    at.toggle(key="unit_toggle").set_value(False).run()
    assert not at.exception
    assert at.selectbox(key="bolt_size").value == 'M10'

def test_optional_toggles():
    """D4: Test that the app runs successfully with all optional toggles enabled."""
    at = AppTest.from_file("app.py")
    at.run(timeout=15)
    
    at.toggle(key="use_washer").set_value(True).run()
    at.toggle(key="thermal_effects").set_value(True).run()
    at.number_input(key="temp_assembly").set_value(20.0).run()
    at.number_input(key="temp_operating").set_value(150.0).run()
    at.number_input(key="embedment_um").set_value(5.0).run()
    
    at.toggle(key="check_fatigue").set_value(True).run()
    
    at.toggle(key="check_stripping").set_value(True).run()
    
    assert not at.exception
''')
