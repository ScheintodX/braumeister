"use strict";

(function($){

	var ctrl = BAG.Ctrl( {} );
	
	var lastMessage = {};

	function loadDevice( $parent, device ) {

		var dId = device.id,
		    dType = device.type,
			dModule = dType.substring( 0,1 ).toUpperCase() + dType.substring( 1 ),
			dControls = dModule + "_Controls"
			;

		console.trace( "LOAD", dId );

		var id = dId,
			$device = $( '<section class="tab device ' + dType + '"/>' )
					.attr( 'id', id )
					.appendTo( $parent )
					.load( dType + ".html", function() {

						var $info = $device.find( '.info' ),
						    $dev = $device.find( '.' + dType ),
							$script = $device.find( 'section.script' )
							;

						console.log( dId, dType, dModule );

						// Tab
						ctrl.put( 'tab_'+id, BAG.Tab( $device, id ) );
						if( $info.length > 0 ) ctrl.put( 'info_'+id, BAG.Info( $info, id ) );
						if( BAG[ dModule ] ) ctrl.put( dType + '_'+id, BAG[ dModule ]( $dev, id ) );
						if( BAG[ dControls ] ) ctrl.put( dType + 'ctrl_'+id, BAG[ dControls ]( $device, id ) );
						if( $script.length ) ctrl.put( 'ctrl_'+id, BAG.Script( $device, id ) );

						// fix for elements to display scripts aren't ready when we get that kind of config/data.
						ctrl.gotData( lastMessage );
					} )
					
	}

	function gotData( data ) {

		// store config for data we receive before controls are loaded.
		if( 'scripts' in data ) {
			lastMessage.scripts = data.scripts;
		}
		if( 'config' in data ) {
			lastMessage.config = data.config;
		}

		if( 'config' in data ) {

			var $main = $('main');

			$main.empty();

			for( var i=0; i < data.config.devices.length; i++ ) {

				var device = data.config.devices[ i ];

				loadDevice( $main, device );
			}
		}

		ctrl.gotData( data );
	}

	function main() {

		var com = BAG.Com
			.onData( gotData )
			.start()
			;

		ctrl.onCom( com.send );
	}

	$(window).keydown(function(e) { if (e.keyCode == 123) debugger; });

	$(main);

})($);
